use crate::PROJECT;
use dioxus::prelude::*;

pub fn Results() -> Element {
    let path = PROJECT.read().clone().unwrap();
    let binding = path.join("training_metrics.png");
    let image_path = binding.to_str().unwrap();

    rsx! {
        img {
                src: "{image_path}",
                //style: "width: px;"
            }
        "in the first graph the accuracy should go up on the graph and the second graph the loss should go down on the graph."
        h1{"making predictions"}
        div {
            h2 { "Making Predictions with a churn Model" }
            p { "This code shows how to load a trained model and predict outcomes using new input data." }
            p{"imports"}
            code{"import numpy as np"}
            p{}
            code{"import tensorflow as tf"}
            ul {
                li {
                    code { "model = tf.keras.models.load_model(\"your_model_path.h5\")" }
                    ": Load a trained model from file."
                }
                li {
                    code { "input_data = np.array([[your input data]], dtype=np.float32)" }
                    ": Prepare your input array."
                }
                li {
                    code { "pred_prob = model.predict(input_data)[0][0]" }
                    ": Predict the probability (between 0 and 1)."
                }
                li {
                    code { "pred_label = 1 if pred_prob > 0.5 else 0" }
                    ": Convert to a binary label (0 or 1)."
                }
            }

            p { "output like this:" }

            code {
                "print(f\"Predicted churn probability: {{pred_prob:.4f}}\")\n\
        print(f\"Predicted churn label: {{pred_label}}\")"
            }

        }
        div {
            h2 { "Image Classification Prediction" }

            h3 { "Python Code:" }
            pre {
                code {
                    "import tensorflow as tf\n"
                    "import numpy as np\n"
                    "from PIL import Image\n\n"
                    "# === CONFIG ===\n"
                    "model_path = 'the file path of the model as a .keras'\n"
                    "image_path = 'file path of the image'\n"
                    "image_size = (180, 180)\n\n"
                    "# === LOAD MODEL ===\n"
                    "model = tf.keras.models.load_model(model_path)\n\n"
                    "# === LOAD AND PREPROCESS IMAGE ===\n"
                    "def preprocess_image(path):\n"
                    "    img = Image.open(path).convert('RGB')\n"
                    "    img = img.resize(image_size)\n"
                    "    img_array = np.array(img) / 255.0\n"
                    "    img_array = np.expand_dims(img_array, axis=0)\n"
                    "    return img_array\n\n"
                    "input_img = preprocess_image(image_path)\n\n"
                    "# === PREDICT ===\n"
                    "prob = model.predict(input_img)[0][0]\n"
                    "label = 1 if prob > 0.5 else 0\n\n"
                    "# === INTERPRET ===\n"
                    "label_str = 'Dog' if label == 1 else 'Cat'\n"
                    "print(f'Predicted: {{label_str}} (probability: {{prob:.4f}})')"
                }
            }

            h3 { "Explanation by Section:" }

            h4 { "Configuration" }
            ul {
                li { "Imports TensorFlow, NumPy, and PIL for image processing." }
                li { "Specifies the model path, image path, and required input size." }
            }
            h4 { "Load Model" }
            p { "Loads the saved model using Keras' load_model function so it can be used for predictions." }

            h4 { "Preprocess Image" }
            ul {
                li { "Opens and converts the image to RGB to ensure 3 color channels." }
                    li { "Resizes the image to match the model's expected input size (180x180)." }
                    li { "Normalizes pixel values to a range of 0 to 1 for better model performance." }
                    li { "Adds an extra batch dimension since the model expects batches of images." }
            }
            h4 { "Predict" }
            ul {
                li { "Runs the model on the preprocessed image to get a probability between 0 and 1." }
                li { "Uses a threshold of 0.5 to convert probability into a binary label (Dog or Cat)." }
            }

            h4 { "Interpret Output" }
            ul {
                li { "Maps the numerical label to a human-readable label: 'Dog' or 'Cat'." }
                li { "Prints the final result along with the prediction probability." }
            }
        }
    }
}
