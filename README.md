# SPIRO Data AI Training Tools

This repository contains code for processing and preparing data from [SPIRO](https://github.com/jonasoh/spiro) - a plant phenotyping platform that captures time-lapse images of seeds growing in Petri plates - for use in AI model training.

## Purpose

The code here helps transform SPIRO-generated images into suitable training data for machine learning models, enabling AI-based analysis of plant growth patterns.

## Development Note

This project was developed primarily using AI-assisted programming tools as part of a course evaluation of AI pair programming capabilities. It serves as both a practical tool and a case study in AI-assisted development.

## Scripts

1. **[`create_coco.py`](create_coco.py)**:
   - Generates COCO JSON files from the training data. It processes images and masks, extracts metadata, and creates annotations in COCO format. It also splits the data into training, validation, and test sets based on seed IDs.

2. **[`germination_inference.py`](germination_inference.py)**:
   - Performs inference on a single image using a pre-trained Roboflow model for seed germination classification. It initializes the Roboflow model, performs prediction, and prints the classification result and confidence.

3. **[`predict_seeds.py`](predict_seeds.py)**:
   - Analyzes seeds in an image using a trained Mask R-CNN model. It loads the model, extracts seed regions based on coordinates, processes the image, and draws predictions on the image. The results are saved to an output file.

4. **[`prepare_spiro_data.py`](prepare_spiro_data.py)**:
   - Processes and crops SPIRO images for AI model training. It reads image files, extracts regions of interest, generates masks, and saves the processed images and masks. It supports both cropped and uncropped image processing.

5. **[`separate_seed_images.py`](separate_seed_images.py)**:
   - Separates seed images based on germination data. It reads germination information from a TSV file, groups images by seed and category, and saves the sorted images into different directories for germinated, ungerminated, and undecided categories.

6. **[`train_pytorch.py`](train_pytorch.py)**:
   - Trains a Mask R-CNN model for seed detection using PyTorch. It loads training and validation datasets, configures the model, and performs training and validation. The best model based on validation loss is saved to a checkpoint file.