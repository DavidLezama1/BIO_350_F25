# BIO_350_F25
Starting our BIO350 assigments repo.

_____________________________________________________________________________

IC Assingment 1: Chapter 1 of Ecology Handbook: Change over Time

• Making a difference equation

• Techanical difficulties, SHOULD be btter from now on

• It works!

_____________________________________________________________________________

IC Assignment 2: Chapter 2 of Ecology Handbook: Duckweed on a Pond: Exponential Growth

• Making an exponential equation to measure how quickly duckeweed covers a pond

• Minimal techanical difficulties, bit better

• Works and introduced matplotlib, and plotting on x and y axis

_____________________________________________________________________________

IC_Assigment 3: Chapter 3 of Ecology Handbook: Throwing Shade I: Logistic Growth

• All information and values come from Chapter 3 of the Handbook of Quantitative Ecology

• This program shows how a plant population grows over time using a logistic growth model. It starts with 12 plants and applies the logistic equation each year to account for growth slowing as the population nears the carrying capacity of 400

• The analysis technique is a discrete-time population model

• The program simulates plant population growth with self-shading and explicit deaths and without

_____________________________________________________________________________

IC Assignment 4: Chapter 4 of Ecology Handbook: Throwing Shade II: Lotka-Volterra Competition

The code shows you how two competing plant species grow and interact in the same plot using logistic growth with self-shading and competition, showing that the colonizer eventually stabilizes at 240 (carrying capacity) while the threatened species goes extinct at year 24. 

• The program sets the initial populations, carrying capacities, number of seeds produced per year, death rates,  and number of years to simulate for both species. 

• The function records the population sizes over time and I made sure to prevent any negative values.

• The simulation runs for  50 years, and the results are stored.

• The model uses a discrete-time logistic growth simulation with Lotka–Volterra competition to analyze species interactions.

_____________________________________________________________________________

IC Assignment 5: Chapter 5 of Ecology Handbook: Rabies Removal: SIR Models

• Uses an SIR model function that simulates rabies spread in a fox population using difference equations for Susceptible (S), Infected (I), and Removed/Dead (R).

• Initializes the population states with given values: 999 susceptible, 1 infected, and 0 dead.

• Iterates daily updates over the time period (40 days), applying the equations for infection (beta variable), death (death rate), and transitions between classes.

• Stores results in arrays so that the daily values of S, I, and R are stored.

• Plots for susceptible, infected, and dead foxes over time

• Prints an answer to the chapter question, showing that rabies spreads and giving the number of foxes dead by day 40 (R[-1]).

• I also answered question given by the teacher about beta variable and death rate changing. 

_____________________________________________________________________________

IC_InclassExample1:

• Made the log of the graph from assignment 2

_____________________________________________________________________________

IC_InclassExample2:

• Imported population data from a CSV file using pandas

• Plotted population size over time (days) using matplotlib, with days on the x-axis, population on the y-axis
and a horizontal reference line at y=276
 
• Calculated average population size between day 17 and day 60 to make the horizontal reference line at y=276

• This analysis uses a density-independent approach because the analysis does not include any factors or equations that account for population regulation by density. It simply summarizes and visualizes population data without modeling how population size affects growth

_____________________________________________________________________________

IC_InclassExample3:

• The code demonstrates two versions of a Lotka–Volterra style competition model for two species (N and M). 

• Each version simulates population growth, competition, and perturbations (extra individuals added at specific time steps). 

• The first model uses logistic growth with death rates, while the second uses the classical competition form with growth rates and competition coefficients. 

• Both models store population sizes over time in lists and then plot trajectories (days vs population) using matplotlib. 

• Example runs are provided with initial values, carrying capacities, and parameters, showing how species interact and respond to competition and sudden introductions.

_____________________________________________________________________________

IC_InclassExample4:

• Imported and installed necessary Python libraries (numpy, scipy, matplotlib, seaborn, pandas).

• Translated an R model (Carcass R file) to Python, defining SICZ and SICZ_gamma compartmental disease models as systems of ODEs.

• Set initial conditions and parameters for different population sizes.

• Solved the ODEs using scipy's odeint for both standard and gamma-distributed models.

• Plotted time series results for Susceptible, Infected, Carcass, and Zoospore compartments for each scenario.

• Each scenario simulated disease spread in large, medium, and small populations, and compared standard versus gamma-distributed infection periods with both slow and fast carcass decay rates.

• Also included a reflection on my experience and the exact steps taken to complete the assignment.