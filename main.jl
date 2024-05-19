using Luxor, Colors, BenchmarkTools, CSV, DataFrames
using Base.Threads
import GeometryTypes, JSON

@enum ViralState susceptible infectious recovered removed asymptomatic
@enum BuildingType house workplace shop

buildingTypes = [house workplace shop]

mutable struct Person
  id::UInt64
  currentPos::Tuple{UInt64, Float64}
  viralState::ViralState
  path::Vector{Tuple{UInt64, Bool}} # (road, direction)
  currentDir::Bool
  infectionTime::Float64
  totalInfectionTime::Float64
  infectionCount::Int64
end

struct Road
  id::UInt64
  pointA::Point
  pointB::Point
end

struct Building
  id::UInt64
  buildingType::BuildingType
  pos::Tuple{UInt64, Float64}
  size::Float64 # ← utilité ?
end

struct Crossing
  id::UInt
  pos::Point
  connectedRoads::Vector{Road}
end

const ∞ = Inf

frameCount = 24 * 60 * 5
dt = 1 / 24

meter = 1 / (111.32 * 1000)  # Conversion factor from meters to degrees at the equator

# simulation constants
const citySize = 100
const roadPointCount = 300
const populationCount = 300000
const buildingCount = 60
const mouvementSpeed = 10 * meter
const infectionProbabilityScale = 2 / 50
const recoverProbability = 97 / 100
const asymptomaticProbability = 2 / 3
const infectionTimeMean = 20.0
const infectionTimeStandardDeviation = 3.0
const recoveredTimeMean = 30.0
const recoveredTimeStandardDeviation = 3.0

# rendering constants
const canvasWidth, canvasHeight = 1000, 1000
const grayColor = Gray(31/255)

𝑤(road::Road) = 1
id() = rand(UInt)

len(v) = v |> x->x.^2 |> sum

roadLength(r::Road) = len(r.pointA - r.pointB) |> sqrt

point(z::Complex) =
  Point(Float64(real(z)), Float64(imag(z)))

Base.:/(a::Point, b::Point) =
  Point(a.x / b.x, a.y / b.y)

function norm(pt::Point)
  vec = [pt.x pt.y]
  sqrt.(vec*vec')[1,1]
end

function getPos(road::Road, point::Point)
  norm(point - road.pointA) / norm(road.pointB - road.pointA)
end

remap(value, start1, stop1, start2, stop2) =
  ((value - start1) / (stop1 - start1)) * (stop2 - start2) + start2

function paired(x; withIndex = false)
  if withIndex
    zip(x[1:end-1], x[2:end], 1:length(x)-2)
  else
    zip(x[1:end-1], x[2:end])
  end
end


function checkPosition((r,d))
  return (r, min(1, max(0, d)))
end

infectionTime() = infectionTimeStandardDeviation * normalRandom() + infectionTimeMean

function infect(p)
  p.viralState = infectious
  p.infectionTime = infectionTime()
  p.totalInfectionTime = p.infectionTime
end

function intersectRoad(roadA::Road, roadB::Road, id::UInt)
  pt(p) = GeometryTypes.Point2f0(p.x, p.y)
  lineA = GeometryTypes.LineSegment(pt(roadA.pointA), pt(roadA.pointB))
  lineB = GeometryTypes.LineSegment(pt(roadB.pointA), pt(roadB.pointB))
  inter, pointₜₑₘₚ = GeometryTypes.intersects(lineA, lineB)

  point = Point(pointₜₑₘₚ[1], pointₜₑₘₚ[2])

  posA = getPos(roadA, point)
  posB = getPos(roadA, point)

  IntersectionPoint(
		    inter, point,
		    roadA, roadB,
		    posA, posB, id
		    )
end


# Function to read the CSV file and convert it to a list of Roads
function read_roads_from_csv(filename)
  df = CSV.read(filename, DataFrame, header=true)
  
  # Extracting the column data from the DataFrame
  x_a_col = df.x_a
  y_a_col = df.y_a
  x_b_col = df.x_b
  y_b_col = df.y_b
  
  # Creating an empty list to store the Roads
  roads = [
    Road(
      id(),
      Point(x_a_col[i], y_a_col[i]),
      Point(x_b_col[i], y_b_col[i])
    )
    for i in 1:size(df, 1)
  ]
  
  return roads
end

println("starting to import the roads")

# Call the function to read the CSV file and obtain the list of Roads
roads = read_roads_from_csv("nantes.csv")

println("got the roads ", length(roads))

function getCrossings(roads::Vector{Road})
  local crossingPoints = Set(pt
			     for road in roads for pt in (road.pointA, road.pointB))
  local connections = Dict{Point,Vector{Road}}(pt => [] for pt in crossingPoints)
  for road in roads
    push!(connections[road.pointA], road)
    push!(connections[road.pointB], road)
  end
  crossings = unique([
		      Crossing(id(), pt, roads)
		      for (pt, roads) in pairs(connections)
		     ])
end

function getPointToCrossingDict(crossings::Vector{Crossing})
  pointToCrossing = Dict{Point,Crossing}()
  for crossing in crossings
    pointToCrossing[crossing.pos] = crossing
  end
  pointToCrossing
end

function isConnected(roads::Vector{Road})
  crossings = getCrossings(roads)
  pointToCrossing = getPointToCrossingDict(crossings)
  visited = Set{Road}()

  function traverse(road::Road)
    if road ∉ visited
      push!(visited, road)

      cA = pointToCrossing[road.pointA]
      cB = pointToCrossing[road.pointB]

      traverse.(cA.connectedRoads)
      traverse.(cB.connectedRoads)
    end
  end

  traverse(first(roads))

  length(visited) == length(roads)
end

distBetween(a,b) = len(b.-a)

function normalRandom()
  (u, v) = (0.0, 0.0)

  while u == 0.0
    u = rand(Float64)
  end

  while v == 0.0
    v = rand(Float64)
  end

  return sqrt(-2.0 * log(u)) * cos(2.0 * pi * v)
end

function randPoint()
  rStandardDeviation = citySize / 3
  r = normalRandom() * rStandardDeviation # + z * citySize / 2

  while r < 0 || r > citySize / 2
    r = normalRandom() * rStandardDeviation # + citySize / 2
  end

  θ = rand() * 2.0 * pi

  return Point(r * cos(θ), r * sin(θ))
end

# Algorithme Barbaroux
function createGraph(R)
  #R = 1.15
  pts = [ randPoint() for i ∈ 1:roadPointCount ] .+ Point(citySize / 2, citySize / 2)

  function shouldAddEdge(i,j)
    m = (pts[i] + pts[j]) / 2
    for k ∈ 1:length(pts)
      if k ≠ i && k ≠ j && R * distBetween(m, pts[i]) ≥ distBetween(m, pts[k])
	      return false
      end
    end
    return true
  end

  return [
   Road(id(), pts[i], pts[j])
   for i in 1:length(pts)
   for j in 1:length(pts)
   if i ≠ j && shouldAddEdge(i,j)
   ]
end

function generatePathFindingMatrices(crossings, roads)
  n = length(crossings)
  dist = fill(∞, (n,n))
  next = fill(-1, (n,n))
  pointToCrossing = getPointToCrossingDict(crossings)
  crossingMap = Dict{UInt,Int}(crossings[i].id => i for i ∈ 1:length(crossings))

  for road in roads
    u = crossingMap[pointToCrossing[road.pointA].id]
    v = crossingMap[pointToCrossing[road.pointB].id]
    w = 𝑤(road)
    dist[u,v] = w
    next[u,v] = v
    dist[v,u] = w
    next[v,u] = u
  end

  for crossing in crossings
    v = crossingMap[crossing.id]
    dist[v,v] = 0
    next[v,v] = v
  end

  for k ∈ 1:n
    @threads for i ∈ 1:n
      for j ∈ 1:n
        if dist[i,j] > dist[i,k] + dist[k,j]
          dist[i,j] = dist[i,k] + dist[k,j]
          next[i,j] = next[i,k]
        end
      end
    end
  end

  dist, next
end

function findPath(u::Int, v::Int, next, dist)
  try
    if next[u,v] == -1
      return []
    end
  catch e
    lock(lk) do
      println(u)
      println(v)
      println(next)
      rethrow()
    end
  end

  d = dist[u,v]

  path = [u]

  while u ≠ v
    u = next[u,v]
    push!(path, UInt(u))
  end

  path, d
end

function crossingListPathGenerator(choosedCrossings, roads, crossings)
  function choosePath(road)
    local (a,b) = road
    for r in roads
      if (r.pointA == crossings[a].pos && r.pointB == crossings[b].pos)
	return (r.id, true)
      end
      if (r.pointA == crossings[b].pos && r.pointB == crossings[a].pos)
	return (r.id, false)
      end
    end
  end

  choosedRoads = zip(choosedCrossings[2:end], choosedCrossings[1:end-1])
  choosedPath = choosePath.(choosedRoads)
  return choosedPath
end

function generatePath(buildings, person::Person, roadMap, pointToCrossingDict, crossingMap, next, dist, roads, crossings)
  getCrossing(pt) = pointToCrossingDict[pt]

  target = rand(buildings)
  targetRoad = roadMap[target.pos[1]]
  targetEnds = getCrossing.([targetRoad.pointA, targetRoad.pointB])
  currentRoad = roadMap[person.currentPos[1]]
  currentEnds = getCrossing.([currentRoad.pointA, currentRoad.pointB])
  paths = [ findPath(crossingMap[b.id], crossingMap[e.id], next, dist) for b in currentEnds for e in targetEnds ]
  choosedCrossings, _ = first(sort(paths; by=last))
  path = crossingListPathGenerator(choosedCrossings, roads, crossings)
  if length(path) == 0
    return generatePath(buildings, person, roadMap, pointToCrossingDict, crossingMap, next, dist, roads, crossings)
  else
    #println(path)
    person.path = path
    _, dir = first(path)
    person.currentDir = dir
  end
end

function randomBuilding(roads, id::UInt64)
  pos = rand(roads).id, rand(Float64)
  size = rand(Float64) * 5 + 1
  buildingType = rand(buildingTypes)
  Building(id, buildingType, pos, size)
end

function randomPerson(roads, crossings, buildings, roadMap, pointToCrossingDict, crossingMap, next, dist, id::UInt64)
  pos = rand(roads).id, rand(Float64)
  person = Person(id, pos, susceptible, [], rand(Bool), 0.0, 0.0, 0)
  generatePath(buildings, person, roadMap, pointToCrossingDict, crossingMap, next, dist, roads, crossings)
  person
end

infectionProbability(distance, δ) = exp(-5 * distance / δ) * infectionProbabilityScale

isInfected(p) = p.viralState == infectious || p.viralState == asymptomatic

function move(person, Δt, buildings, roadMap, pointToCrossingDict, crossingMap, next, dist, roads, crossings)
  r, d = person.currentPos
  Δd = mouvementSpeed * Δt / roadLength(roadMap[r])
  if person.currentDir # A →  B
    if d + Δd < 1.0
      person.currentPos = (r, d + Δd) |> checkPosition
    else # d + Δd ≥ 1.0
      if length(person.path) == 0
	      generatePath(buildings, person, roadMap, pointToCrossingDict, crossingMap, next, dist, roads, crossings)
      end
	
      next, dir = popfirst!(person.path)
      person.currentDir = dir
      d = d + Δd - 1.0
      person.currentPos = (next, dir ? d : (1 - d)) |> checkPosition
    end
  else # B →  A
    if d - Δd ≥ 0.0
      person.currentPos = (r, d - Δd) |> checkPosition
    else # d + Δd < 0.0
      if length(person.path) == 0
	      generatePath(buildings, person, roadMap, pointToCrossingDict, crossingMap, next, dist, roads, crossings)
      end
	
      next, dir = popfirst!(person.path)
      person.currentDir = dir
      d = d - Δd + 1.0
      person.currentPos = (next,!dir ? d : (1 - d)) |> checkPosition
    end
  end
end

function updateCity(Δt, population, buildings, roadMap, pointToCrossingDict, crossingMap, next, dist, roads, crossings, δ, infectedCount)
  Rvalues = []

  for person ∈ population
    move(person, Δt, buildings, roadMap, pointToCrossingDict, crossingMap, next, dist, roads, crossings)

    if person.viralState == susceptible
      for infected_person ∈ population
        if isInfected(infected_person)
          d = distBetween(at(roadMap, person.currentPos), at(roadMap, infected_person.currentPos))
          if rand() < infectionProbability(d, δ)
            infect(person)
            infected_person.infectionCount = infected_person.infectionCount + 1
            infectedCount += 1
          end
        end
      end
    elseif person.viralState == infectious
      prop = (person.totalInfectionTime - person.infectionTime) / person.totalInfectionTime

      if prop > 0.1
        Rval = person.infectionCount / prop
        push!(Rvalues, Rval)
      end

      person.infectionTime -= Δt
      if person.infectionTime <= 0
	      person.viralState = removed
        infectedCount -= 1
      end
    end
  end

  if length(Rvalues) == 0
    return infectedCount, 0
  else
    return infectedCount, mean(Rvalues)
  end
end

#########################################
############### RENDERING ###############
#########################################

function drawCity(
  roads::Array{Road,1},
  population::Array{Person,1},
  buildings::Array{Building,1};
  half::Bool = false)

  w = canvasWidth
  h = canvasHeight

  if half
    w, h = w/2, h/2
  end
  
  origin()
  draw(half)
  draw.(roads, half)
  draw.(buildings, half)
  draw.(population, half)
end

function draw(half::Bool)
  setcolor(grayColor)
  if half
    rect(-canvasWidth/4, -canvasHeight/4, canvasWidth/2, canvasHeight/2, action=:fill)
  else
    rect(-canvasWidth/2, -canvasHeight/2, canvasWidth, canvasHeight, action=:fill)
  end
end

function draw(person::Person, half::Bool)
  chooseColor(person.viralState)
  circle(remapCoords(getCoords(person), half), 2, action=:fill)
end

function draw(road::Road, half::Bool)
  setcolor(Gray(0.5))
  arrow(remapCoords(road.pointA, half), remapCoords(road.pointB, half))#, action=:stroke)
end

function draw(building::Building, half::Bool)
  setcolor(colorant"#ff00ff")
  square(remapCoords(getCoords(building), half), 5)
end

function chooseColor(state::ViralState)
  color = if state == infectious
      colorant"#ff6347"
    elseif state == asymptomatic
      colorant"#DC143C"
    elseif state == susceptible
      colorant"#41D1CC"
    elseif state == recovered
      colorant"#9ACD32"
    elseif state == removed
      colorant"#696969"
    else
      colorant"#ffff00"
    end
  setcolor(color)
end

function square(center::Point, r)
  rect(center.x-r, center.y-r, 2r, 2r, action=:fill)
end

function remapCoords(pt::Point, half::Bool)
  padding = 0.9
  if half
    padding * remap(
      pt,
      Point(0, 0),
      Point(citySize, citySize),
      -Point(canvasWidth, canvasHeight) / 4,
      Point(canvasWidth, canvasHeight) / 4
    )
  else
    padding * remap(
      pt,
      Point(0, 0),
      Point(citySize, citySize),
      -Point(canvasWidth, canvasHeight) / 2,
      Point(canvasWidth, canvasHeight) / 2
    )
  end
end

at(roadMap,road::UInt64, displace::Float64) = at(roadMap,roadMap[road], displace)
at(roadMap,road::Road, displace::Float64) = displace * road.pointA + (1 - displace) * road.pointB
at(roadMap,pos::Tuple{UInt64,Float64}) = at(roadMap,pos[1], pos[2])

getCoords(roadMap,person::Person) = at(roadMap,person.currentPos)
getCoords(roadMap,building::Building) = at(roadMap,building.pos)


#########################################
############### UTILITIES ###############
#########################################

function idx(vec::Point)
  rA = [ road for road in roads if road.pointA == vec ]
  if rA == [ ]
    rB = [ road for road in roads if road.pointB == vec ]
    first(rB).id * 2 + 2
  else
    first(rA).id * 2 + 1
  end
end

roadCount = length(roads)
crossings = getCrossings(roads)
println("Got crossings")
pointToCrossingDict = getPointToCrossingDict(crossings)
println("Got dict")
# dist, next = generatePathFindingMatrices(crossings, roads)

dist = reduce(hcat, JSON.Parser.parsefile("dist.json"))
next = reduce(hcat, JSON.Parser.parsefile("next.json"))

println("Got matrices")
crossingMap = Dict{UInt,Int}(crossings[i].id => i for i ∈ 1:length(crossings))
roadMap = Dict{UInt,Road}(r.id => r for r in roads)
println("Got all the data needed to start simulation...")

function simulate(δ)
  local buildings = [ randomBuilding(roads, UInt64(id)) for id ∈ 1:buildingCount ]
  local population = [ randomPerson(roads, crossings, buildings, roadMap, pointToCrossingDict, crossingMap, next, dist, UInt64(id)) for id ∈ 1:populationCount ]

  infect(rand(population))
  infect(rand(population))
  infect(rand(population))

  local data = zeros(frameCount)
  local Rvalues = zeros(frameCount)
  local infectedCount = 3

  for i in 1:frameCount
    data[i], Rvalues[i] = updateCity(1, population, buildings, roadMap, pointToCrossingDict, crossingMap, next, dist, roads, crossings, δ, infectedCount)
    infectedCount = data[i]
    println("# ", 100*i/frameCount, "%")
  end

  #anim = Movie(canvasWidth, canvasHeight, "out", 1:frameCount)
  #animate(anim, Scene(anim, drawScene, 1:frameCount); creategif=true, pathname="./out" * string(z) * ".gif")

  data, Rvalues
end

n = 20

f1 = open("output-7a-nantes.csv", "w")
f2 = open("output-7b-nantes.csv", "w")

a = zeros(frameCount)
b = zeros(frameCount)

lk3=ReentrantLock()
lk4=ReentrantLock()

@threads for i in 1:n
  println("Started ", i)
  v, R = simulate(0.5 * meter)
  lock(lk3) do
    a[:] = a .+ (v ./ n)
  end
  lock(lk4) do
    b[:] = b .+ (R ./ n)
  end
  println("Done ", i)
end

for x in a
  print(f1, x, ',')
end
println(f1, "")

for x in b
  print(f2, x, ',')
end
println(f2, "")

close(f1)
close(f2)
