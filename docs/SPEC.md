# EVOMORPH — Build an Artificial Evolutionary Robotics Research Laboratory

> This is the original project brief, kept verbatim as the source of truth for scope.
> The brief was cut off mid-sentence at the end of section 14; nothing after that point exists.

You are an expert robotics research engineer, computational scientist, evolutionary computation researcher, and senior software architect.

Your task is to build a sophisticated, fully functional, software-only evolutionary robotics laboratory called **EVOMORPH**.

This is not a conventional application. It is an experimental research platform for investigating how entirely artificial creatures can evolve their own physical morphologies and locomotion strategies through simulated natural selection.

I have a capable laptop but absolutely NO physical robotics hardware. Everything must run in software using real physics simulation.

I also have a limited four-hour development window for this initial build. Prioritize a functioning, technically sound research prototype over excessive planning, architecture discussions, or superficial polish.

Do not ask me questions. Make sensible engineering decisions, inspect my machine, and start implementing.

---

# 1. THE RESEARCH OBJECTIVE

Build a virtual world in which artificial creatures evolve through generations.

Every creature has:
- A physical body.
- A configurable number of limbs.
- Articulated joints.
- A set of actuators.
- A controller responsible for movement.
- A genetic representation that can be mutated and reproduced.

Initially, creatures should be randomly generated and largely incapable of locomotion.

Over successive generations, evolutionary algorithms should discover increasingly effective body structures and movement patterns.

The central research question is:

**Does jointly evolving robot morphology and locomotion controllers produce more robust and behaviorally diverse locomotion than evolving controllers for a predefined robot body?**

I want this to feel like artificial life research combined with evolutionary robotics.

The system should generate creatures that I did not explicitly design, discover locomotion strategies that I did not explicitly program, and produce measurable experimental results.

Do NOT simply create a collection of predetermined robot models with different colors.

The physical structures and locomotion parameters must genuinely evolve.

# 2. TECHNOLOGY STACK

Use the following technologies:

### Physics simulation
- MuJoCo
- Python native MuJoCo bindings
- MJCF generated programmatically

MuJoCo must be responsible for actual physics: gravity, rigid body dynamics, joint constraints, actuation, ground contact, friction, and collisions.

Do not substitute a custom kinematic animation or approximate movement system for real physics.

### Evolutionary computation
- Python
- NumPy
- SciPy where useful
- Custom genetic algorithm
- MAP-Elites quality-diversity optimization

### Controllers
Start with low-dimensional Central Pattern Generator (CPG) controllers.

These should produce periodic joint movements using evolved frequencies, amplitudes, and phase relationships.

Design the controller interface so that neural network policies can be added later.

### Visualization
- MuJoCo rendering and interactive viewer
- Matplotlib
- Plotly if useful
- Streamlit for a lightweight research dashboard, if the core system is working

### Engineering
- pytest
- YAML experiment configurations
- JSON genome and experiment serialization
- Git-friendly repository organization

Use Python 3.11 or 3.12.

Everything must be CPU-compatible. Do not assume CUDA or an NVIDIA GPU exists.

Do not install ROS 2, Isaac Sim, Omniverse, or other unnecessarily heavy infrastructure for this first version.

---

# 3. THE ROBOT MORPHOLOGY GENERATOR

This is one of the most important components.

Create a procedural robot generator that takes a genetic representation and creates a valid MuJoCo MJCF model.

Creatures should have a central torso with symmetrical pairs of articulated limbs.

Initially support:

### Number of limbs
Allow:
- 2 limbs
- 4 limbs
- 6 limbs
- 8 limbs

### Morphological parameters

The genome should encode at least:

- Number of limb pairs.
- Torso length.
- Torso width.
- Torso height.
- Torso mass scale.
- Upper limb length.
- Lower limb length.
- Limb thickness.
- Foot radius.
- Overall limb mass.
- Whether limbs have one or two articulated segments.
- Joint range limits.

Use sensible physical parameter ranges to avoid generating invalid or physically meaningless models.

Support variations in both body proportions and limb structures.

### Procedural MJCF generation

Generate each creature's MuJoCo model from its genome.

Every model must have:
- A free root joint.
- A torso.
- Symmetrical limbs.
- Appropriate revolute joints.
- Actuators.
- Contact-enabled feet.
- Realistic mass and inertia properties.
- Collision geometry.
- Stable initial conditions.

Generate unique names for bodies, joints, and actuators.

You must validate generated MJCF before evaluating a creature.

Handle invalid genotypes gracefully.

Implement a morphology viewer so I can inspect any creature independently.

IMPORTANT: The robot generator must support different numbers of actual joints. Do not build one hardcoded quadruped and merely scale its body.

---

# 4. EVOLVABLE LOCOMOTION CONTROLLER

Initially implement a Central Pattern Generator controller.

Each limb should receive an oscillatory joint target.

For example:

hip_target = A_hip * sin(2π * frequency * t + phase)

knee_target = A_knee * sin(2π * frequency * t + phase + knee_offset)

The controller should support different gait coordination strategies.

Implement:
- Alternating gait.
- Diagonal gait.
- Tripod gait.
- Metachronal wave gait.

The controller parameters must also be part of the creature's genome.

Include:
- Oscillation frequency.
- Hip amplitude.
- Knee amplitude.
- Knee phase offset.
- Gait coordination pattern.
- Per-limb phase variation.

The controller must dynamically adapt to the number of limbs generated by the morphology.

Eventually, I want to experiment with evolving neural network controllers, but do not sacrifice the working CPG implementation to build an unfinished neural network.

Create a clean controller interface that allows future controller implementations to use the same simulation environment.

---

# 5. THE EVOLUTION ENGINE

Implement an actual evolutionary algorithm.

Each creature represents an individual.

A generation consists of:
1. Generating an initial population.
2. Constructing physical robots from their genomes.
3. Simulating each individual.
4. Measuring fitness.
5. Selecting parents.
6. Performing crossover.
7. Applying mutations.
8. Preserving elite individuals.
9. Creating a new generation.

### Genetic operations

Implement:

**Mutation**
- Continuous parameter mutation.
- Random parameter resets.
- Discrete morphology mutations.
- Gait pattern mutations.
- Limb count changes.
- Segment configuration changes.

**Crossover**
- Combine morphological characteristics of two parent creatures.
- Combine controller parameters.
- Ensure children always produce valid genotypes.

**Selection**
- Tournament selection or rank-based selection.
- Elitism.
- Diversity-preserving mechanisms.

Avoid designs where evolution quickly converges to nearly identical creatures without exploration.

### Population configuration

The system must support configurable:
- Population size.
- Number of generations.
- Mutation probability.
- Mutation magnitude.
- Crossover probability.
- Elite count.
- Simulation duration.
- Random seed.

Create a quick experimental configuration that runs efficiently on a laptop.

Do not assume every simulation evaluation can use a GPU.

---

# 6. FITNESS FUNCTION

Design a multi-component fitness function.

The initial goal is locomotion.

Measure:

1. Forward displacement.
2. Average forward velocity.
3. Distance traveled.
4. Survival time.
5. Body stability.
6. Excessive rotation.
7. Lateral displacement.
8. Energy consumption.
9. Locomotion smoothness.

Reward creatures that travel forward while remaining stable and minimizing unnecessary energy expenditure.

Penalize:
- Falling over.
- Remaining stationary.
- Excessive lateral movement.
- Excessive rotation.
- Unstable movement.
- Excessive actuator effort.

Do not reward speed alone.

A creature that launches itself forward, immediately collapses, or simply spins should not become the best creature.

Record the individual metric values separately from the combined fitness score.

Implement fitness normalization carefully so individual terms do not unintentionally dominate the objective.

Most importantly, make sure the fitness function actually rewards functional locomotion.

---

# 7. QUALITY-DIVERSITY EVOLUTION

This is where I want the project to go beyond a conventional genetic algorithm.

Implement MAP-Elites or a comparable quality-diversity algorithm.

Rather than maintaining only one best creature, maintain an archive of creatures with different behavioral characteristics.

For the initial implementation, create a two-dimensional archive using:

- Forward speed.
- Energy efficiency.

Each archive cell stores the highest-quality individual discovered for that behavioral region.

The archive should preserve different kinds of successful creatures rather than allowing one particular morphology to dominate everything.

Display:
- Archive occupancy.
- Creature diversity.
- Best individual in each cell.
- Fitness distribution.
- Morphological diversity.
- Evolutionary progress.

Implement genetic mutation and exploration mechanisms that can introduce new structures and behaviors into the archive.

I want to be able to inspect the entire collection of successful evolved creatures.

---

# 8. ENVIRONMENT GENERATION

Start with a flat ground environment.

Once locomotion works, implement additional environments:

### Environment A: Flat terrain
A basic locomotion benchmark.

### Environment B: Sloped terrain
Inclined surfaces at different angles.

### Environment C: Uneven terrain
Randomly positioned terrain variations and low obstacles.

### Environment D: Variable friction
Different ground friction coefficients.

### Environment E: Randomized terrain
A combination of slopes, obstacles, and different friction values.

Each creature should be evaluated in configurable environments.

Create the ability to evaluate the same evolved creature across multiple randomized terrain seeds.

Distinguish between training environments and held-out testing environments.

I want to measure whether creatures evolve strategies that generalize beyond the exact environment they were trained in.

---

# 9. THE RESEARCH EXPERIMENT

The central experiment should compare three approaches.

### Experiment A: Controller-only evolution

Use one predefined quadruped morphology.

Only evolve its controller.

### Experiment B: Morphology-only evolution

Allow physical morphology to evolve while keeping controller parameters fixed or drawn from a controlled baseline.

### Experiment C: Joint morphology-controller evolution

Allow both the physical body and locomotion controller to evolve.

Where practical, include MAP-Elites as an additional comparison against conventional single-objective selection.

Use equal evaluation budgets and reproducible random seeds.

Compare the results on:
- Training terrain performance.
- Unseen terrain performance.
- Mean locomotion speed.
- Energy efficiency.
- Stability.
- Failure rate.
- Morphological diversity.
- Behavioral diversity.
- Generalization performance.

Create plots that make these comparisons clear.

The key question is whether joint evolution produces more diverse and robust locomotion than the individual approaches.

Do not assume the hypothesis is correct. The experiments should be capable of showing that one approach performs worse.

---

# 10. VISUALIZATION: MAKE THIS FEEL LIKE AN ARTIFICIAL LIFE LAB

The project should be visually impressive.

I want to watch artificial creatures emerge and evolve.

Create a research dashboard with the following sections.

### A. Evolution viewer

Display:
- Current generation.
- Number of individuals evaluated.
- Best fitness.
- Population average fitness.
- Current mutation rate.
- Archive occupancy.

Show the best creature from each generation.

### B. Creature inspection

Allow selecting an individual and viewing:
- 3D body structure.
- Number of limbs.
- Joint configuration.
- Physical proportions.
- Controller parameters.
- Fitness measurements.
- Parent identifiers.
- Generation of origin.

Allow the user to launch an interactive simulation of the selected creature.

### C. MAP-Elites archive

Visualize the two-dimensional archive.

Allow selecting an occupied cell to inspect its representative creature.

### D. Evolution statistics

Plot:
- Best fitness per generation.
- Mean fitness per generation.
- Fitness distribution.
- Morphology diversity.
- Number of successful creatures.
- Archive occupancy.
- Energy efficiency.
- Average displacement.

### E. Creature comparison

Allow two or more creatures to be selected and compared.

Display their physical structures and locomotion statistics.

If practical, allow their simulation videos to be viewed side by side.

### F. Generation history

Preserve champions from every generation so their development can be inspected later.

Do not just show an animation of randomly generated robots.

Every visualization must correspond to an actual genome and actual simulation results.

---

# 11. DATA AND EXPERIMENT MANAGEMENT

Every experiment must be reproducible.

Each run should save:

- Experiment configuration.
- Random seed.
- Generation number.
- Population fitness.
- Individual genomes.
- Parent relationships.
- Morphology parameters.
- Controller parameters.
- Simulation metrics.
- MAP-Elites archive.
- Best individual.
- Training and evaluation times.

Create a clean run directory, for example:

    results/
        run_001/
            config.yaml
            metrics.csv
            generations/
            genomes/
            archive/
            videos/
            plots/
            best_genome.json
            summary.md

The system must support loading a previous genome and reproducing its simulation.

Ensure experiment results are not silently overwritten.

---

# 12. PROJECT STRUCTURE

Use a clean, modular Python architecture.

Suggested structure:

    evomorph/
        __init__.py
        config.py

        morphology/
            genome.py
            generator.py
            validator.py

        controllers/
            base.py
            cpg.py

        simulation/
            environment.py
            evaluator.py
            metrics.py

        evolution/
            genetic_algorithm.py
            operators.py
            selection.py
            map_elites.py
            archive.py

        experiments/
            baselines.py
            robustness.py
            evaluation.py

        visualization/
            viewer.py
            renderer.py
            plots.py
            dashboard.py

        utils/
            serialization.py
            logging.py
            random_state.py

    configs/
        quick.yaml
        research.yaml

    tests/
    scripts/
    results/
    docs/

You may improve this structure if you have a compelling engineering reason, but preserve clear separation between morphology, controllers, simulation, evolution, and visualization.

---

# 13. COMMAND-LINE INTERFACE

I want simple commands to operate the entire system.

Examples:

Start evolution:

    python -m evomorph evolve --config configs/quick.yaml

Run a longer experiment:

    python -m evomorph evolve --config configs/research.yaml

Inspect a genome:

    python -m evomorph inspect --genome results/run_001/best_genome.json

Evaluate a genome:

    python -m evomorph evaluate --genome results/run_001/best_genome.json

Run baseline comparisons:

    python -m evomorph experiment --type comparison

Generate research plots:

    python -m evomorph analyze --run results/run_001

Launch dashboard:

    streamlit run evomorph/visualization/dashboard.py

Adapt the actual command structure as necessary, but make it easy to reproduce experiments.

---

# 14. TESTING AND SCIENTIFIC VALIDATION

This is extremely important.

Do not declare the project functional merely because the code runs.

Implement tests for:

1. Genome generation.
2. Genome mutation.
3. Genome crossover.
4. MJCF generation.
5. MuJoCo model compilation.
6. Different limb configurations.
7. Controller output dimensions.
8. Physics simulation.
9. Fitness calculations.
10. Genome serialization.
11. Experiment reproducibility.
12. MAP-Elites insertion and retrieval.

Run actual simulations.

Inspect whether the generated creatures can move.

If they cannot move, investigate:
- Incorrect joint axes.
- Incorrect initial body positioning.
- Insufficient ground contact.
- Poor gait phases.
- Insufficient actuator strength.
- Incorrect actuator control.
- Unstable physical proportions.
- Bad fitness incentives.

Fix the underlying issue rather than artificially manipulating fitness scores.

The system must demonstrate real physical locomotion before spending substantial

*(the brief ends here, cut off mid-sentence)*
