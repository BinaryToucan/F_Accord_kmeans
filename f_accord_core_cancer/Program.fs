//#I @"./packages"
//#r @"FSharp.Data.2.3.2/lib/net40/FSharp.Data.dll"
//#r @"Accord.3.4.0/lib/net45/Accord.dll"
//#r @"Accord.MachineLearning.3.4.0/lib/net45/Accord.MachineLearning.dll"
//#r @"Accord.Math.3.4.0/lib/net45/Accord.Math.Core.dll"
//#r @"Accord.Math.3.4.0/lib/net45/Accord.Math.dll"
//#r @"Accord.Statistics.3.4.0/lib/net45/Accord.Statistics.dll"

open System
open Accord.MachineLearning
open Accord.MachineLearning.VectorMachines.Learning
open FSharp.Data

open Accord.Statistics.Models.Regression
open Accord.Statistics.Models.Regression.Fitting
open Accord.DataSets
open Accord.MachineLearning.DecisionTrees

let iris = new Iris()
let inputs = iris.Instances
let outputs = iris.ClassLabels
let kmeans = new BalancedKMeans(3, MaxIterations = 100)
let clusters = kmeans.Learn(inputs)
let labels = clusters.Decide(inputs)
let testIris = [0;36;48]


printf "\n"
printf "Number\t Prediction\t              Label \n"
for i in 0 .. inputs.Length - 1 do
    printf "  %d\t |%s\t         |         (%s)\n" i iris.ClassNames.[labels.[i]] iris.ClassNames.[outputs.[i]]