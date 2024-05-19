# Epidemic Simulator.

Simulating epidemic propagation in a city environment.
This project uses a multiagents system (an agent represents a person moving in a city).
The disease is spread between inhabitants of this city.

To test this model, two environments are available:
- a randomly generated city environment (with the `model-testing.jl` file);
- and a real-world city environment (with the `main.jl` file).

The randomly generated one was mainly used to test the model's correctness.

This project wasn't _really_ meant to be reused, but if you are curious, here are some notes about the files in this project.
If you prefer a more high-level overview of this project and its results, you can look at this [slideshow](https://drive.google.com/file/d/1tAqkL5aptl6dwVY5IfMYFGZb9Zuk8pgb/view) (in French).


In the beginning of the `main.jl` file, you may change the parameters used for the simulations:
```julia
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
```
(The same parameters exists in the `model-testing.jl` file.)

In order for the main program to run, it requires a "map" of the city.
This is what `format.jl` does: it converts a GeoJSON file (for example, one for the French city *Lyon*) to a "map" of this city, with the correct format used by the main file.

To check the generated "map", you can use `show.py` to see the roads of this map.

The main program will start by generating the Floyd-Warshall matrices (and caching them in a file).
Then, it'll start the simulation, save the correct indicators and export this data to a CSV file.
Parallelization is used to save time (most of the computation is done in parallel, and mutex are used to avoid issues with this parallelization).

**WARNING.** Simulations take a long time to run. For me, running all my simulations took about a week and a half on a pretty powerful computer (running 24/7). Running the `main.jl` file took between a day to a day and a half on the computer I used. This is mostly due to the _size_ of the city: for example, getting the Floyd-Warshall matrices has a time complexity $\mathrm{O}(n^3)$ and $n$ is around $20,000$!

Here are some examples of simulations:

![Animation 1](anim1.gif)
![Animation 2](anim2.gif)

If some people seem to disappear at the end of roads, do not worry: this is only a visual bug where people seem to go in the opposite direction. I chose to not fix this, as I turned off the simulations after that (it helps a lot to make the program more efficient).

The `lyon.geojson` was taken from Lyon's _open data_ website. The `LICENSE` file does not apply to it.
