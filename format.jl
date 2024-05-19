using JSON, Luxor

struct Road
    id::UInt64
    pointA::Point
    pointB::Point
end

struct Crossing
    id::UInt
    pos::Point
    connectedRoads::Vector{Road}
end

id() = rand(UInt)

distSr(p) = p.x * p.x + p.y * p.y

function convert_to_julia_type(geojson_file)
    geojson = JSON.parsefile(geojson_file)
    roads = Road[]
    crossings = Crossing[]
    
    feature_collection = geojson["features"]
    
    for feature in feature_collection
        geometry = feature["geometry"]
        
        if geometry["type"] == "LineString"
            coordinates = [ round_longlat.(x) for x in geometry["coordinates"] ]
            
            pointA = Point(coordinates[1][1], coordinates[1][2])
            pointB = Point(coordinates[end][1], coordinates[end][2])

            if distSr(pointA - pointB) <= (2.6 / 3600) ^ 2
                continue
            end
            
            road = Road(id(), pointA, pointB)
            push!(roads, road)
        end
    end
    
    for i in 1:length(roads)
        roadA = roads[i]
        
        for j in (i+1):length(roads)
            roadB = roads[j]
            intersection = find_intersection(roadA, roadB)
            
            if intersection !== nothing
                crossing_id = length(crossings) + 1
                connected_roads = [roadA, roadB]
                
                crossing = Crossing(crossing_id, intersection, connected_roads)
                push!(crossings, crossing)
            end
        end
    end
    
    return roads, crossings
end

function round_longlat(longitude, precision_meters = 78)
    degrees_per_meter = 1 / (111.32 * 1000)  # Conversion factor from meters to degrees at the equator
    precision_degrees = precision_meters * degrees_per_meter
    rounded_longitude = round(longitude / precision_degrees) * precision_degrees
    return rounded_longitude
end

function find_intersection(roadA::Road, roadB::Road)
    xdiff = (roadA.pointA.x - roadA.pointB.x, roadB.pointA.x - roadB.pointB.x)
    ydiff = (roadA.pointA.y - roadA.pointB.y, roadB.pointA.y - roadB.pointB.y)
    
    det = xdiff[1] * ydiff[2] - xdiff[2] * ydiff[1]
    
    if det == 0
        return nothing
    end
    
    d = (roadA.pointA.x * roadA.pointB.y - roadA.pointA.y * roadA.pointB.x, roadB.pointA.x * roadB.pointB.y - roadB.pointA.y * roadB.pointB.x)
    x = (d[1] * xdiff[2] - d[2] * xdiff[1]) / det
    y = (d[1] * ydiff[2] - d[2] * ydiff[1]) / det
    
    intersection = Point(x, y)
    
    if on_segment(roadA, intersection) && on_segment(roadB, intersection)
        return intersection
    else
        return nothing
    end
end

function on_segment(road::Road, point::Point)
    min_x = min(road.pointA.x, road.pointB.x)
    max_x = max(road.pointA.x, road.pointB.x)
    min_y = min(road.pointA.y, road.pointB.y)
    max_y = max(road.pointA.y, road.pointB.y)
    
    return min_x <= point.x <= max_x && min_y <= point.y <= max_y
end


function split_road_segments(roads, crossings)
    split_roads = Road[]
    
    for road in roads
        segments = split_road(road, crossings)
        
        for segment in segments
            push!(split_roads, segment)
        end
    end
    
    return split_roads
end

function split_road(road, crossings)
    segments = Road[]
    
    start_point = road.pointA
    end_point = road.pointB
    
    for crossing in crossings
        intersection = find_intersection(start_point, end_point, crossing)
        
        if intersection !== nothing
            if intersection == start_point || intersection == end_point
                continue
            end
            
            push!(segments, Road(road.id, start_point, intersection))
            start_point = intersection
        end
        
        if start_point == end_point
            break
        end
    end
    
    if start_point != end_point
        push!(segments, Road(road.id, start_point, end_point))
    end
    
    return segments
end

function find_intersection(start_point, end_point, crossing)
    for connected_road in crossing.connectedRoads
        intersection = intersect_segments(start_point, end_point, connected_road.pointA, connected_road.pointB)
        if intersection !== nothing
            return intersection
        end
    end
    
    return nothing
end

function intersect_segments(p1, p2, p3, p4)
    det = (p4.y - p3.y) * (p2.x - p1.x) - (p4.x - p3.x) * (p2.y - p1.y)
    
    if det == 0
        return nothing
    end
    
    ua = ((p4.x - p3.x) * (p1.y - p3.y) - (p4.y - p3.y) * (p1.x - p3.x)) / det
    ub = ((p2.x - p1.x) * (p1.y - p3.y) - (p2.y - p1.y) * (p1.x - p3.x)) / det
    
    if 0 <= ua <= 1 && 0 <= ub <= 1
        intersection_x = round_longlat(p1.x + ua * (p2.x - p1.x))
        intersection_y = round_longlat(p1.y + ua * (p2.y - p1.y))
        
        return Point(intersection_x, intersection_y)
    else
        return nothing
    end
end

roads, crossings = convert_to_julia_type("lyon.geojson")

println(length(roads), " ", length(crossings))

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

    println(length(visited))

    length(visited) == length(roads)
end

function getConnected(roads::Vector{Road})
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
    
    visited
end

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

split_roads = split_road_segments(roads, crossings) |> getConnected

println(length(split_roads))


out = open("lyon.csv", "w")

println(out, "x_a,y_a,x_b,y_b")

for road in split_roads
    println(out, road.pointA.x, ',', road.pointA.y, ',', road.pointB.x, ',', road.pointB.y)
end

close(out)
