using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace CensusIncomeAnalysis
{
    // Define the data class for census features
    public class CensusData
    {
        [LoadColumn(0)]
        public float Age { get; set; }

        [LoadColumn(1)]
        public string Workclass { get; set; }

        [LoadColumn(2)]
        public string Education { get; set; }

        [LoadColumn(3)]
        public string MaritalStatus { get; set; }

        [LoadColumn(4)]
        public string Occupation { get; set; }

        [LoadColumn(5)]
        public string Relationship { get; set; }

        [LoadColumn(6)]
        public string Race { get; set; }

        [LoadColumn(7)]
        public string Sex { get; set; }

        [LoadColumn(8)]
        public float CapitalGain { get; set; }

        [LoadColumn(9)]
        public float CapitalLoss { get; set; }

        [LoadColumn(10)]
        public float HoursPerWeek { get; set; }

        [LoadColumn(11)]
        public string NativeCountry { get; set; }

        [LoadColumn(12), ColumnName("Label")]
        public string Income { get; set; } //  "<=50K" or ">50K"
    }

    // Define the class for predictions
    public class CensusPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedIncome { get; set; } // "<=50K" or ">50K"

        [ColumnName("Score")]
        public float Probability { get; set; } // Probability of the predicted outcome
    }

    class Program
    {
        static void Main(string[] args)
        {
            // 1. Load the data
            MLContext mlContext = new MLContext();
            string dataPath = "census_data.csv"; // Replace with your data file path
            IDataView dataView = mlContext.Data.LoadFromTextFile<CensusData>(dataPath, hasHeader: true, separatorChar: ',');

            // 2. Define the training pipeline
            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding("WorkclassEncoded", "Workclass")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("EducationEncoded", "Education"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("MaritalStatusEncoded", "MaritalStatus"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("OccupationEncoded", "Occupation"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("RelationshipEncoded", "Relationship"))

                .Append(mlContext.Transforms.Categorical.OneHotEncoding("RaceEncoded", "Race"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("SexEncoded", "Sex"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("NativeCountryEncoded", "NativeCountry"))
                .Append(mlContext.Transforms.Concatenate("Features", "Age", "WorkclassEncoded", "EducationEncoded", "MaritalStatusEncoded", 
                                                      "OccupationEncoded", "RelationshipEncoded", "RaceEncoded", "SexEncoded", "CapitalGain", 
                                                      "CapitalLoss", "HoursPerWeek", "NativeCountryEncoded"))
                .Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Score", "Score"));

            // 3. Train the model
            ITransformer model = pipeline.Fit(dataView);

            // 4. Create a prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<CensusData, CensusPrediction>(model);

            // 5. Make a prediction
            CensusData newPerson = new CensusData()
            {
                // Example person data
                Age = 35,
                Workclass = "Private",
                Education = "HS-grad",
                MaritalStatus = "Married-civ-spouse",
                Occupation = "Craft-repair",
                Relationship = "Husband",
                Race = "White",
                Sex = "Male",
                CapitalGain = 0,
                CapitalLoss = 0,
                HoursPerWeek = 40,
                NativeCountry = "United-States"
            };

            CensusPrediction prediction = predictionEngine.Predict(newPerson);

            // 6. Display the prediction
            Console.WriteLine($"Predicted Income: {prediction.PredictedIncome}");
            Console.WriteLine($"Probability: {prediction.Probability}");

            Console.ReadKey();
        }
    }
}
