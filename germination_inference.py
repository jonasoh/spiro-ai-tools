import os
import argparse
from roboflow import Roboflow

# this script is just a simple wrapper for testing out roboflow inference


def classify_image(image_path):
    # Initialize Roboflow
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))

    # Load the model
    project = rf.workspace().project("seed-germination-rrkg0")
    model = project.version(1).model

    # Perform inference
    result = model.predict(image_path).json()

    # Extract classification result
    if result and "predictions" in result and len(result["predictions"]) > 0:
        prediction = result["predictions"][0]["predictions"][0]
        print(prediction)
        class_name = prediction["class"]
        confidence = prediction["confidence"]
        return class_name, confidence
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Classify a single image using the seed-germination-rrkg0 Roboflow model"
    )
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()

    class_name, confidence = classify_image(args.image_path)

    if class_name and confidence:
        print(f"Classification: {class_name}")
        print(f"Confidence: {confidence:.2f}")
    else:
        print("No classification result found")


if __name__ == "__main__":
    main()
