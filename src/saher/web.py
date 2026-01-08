import gradio as gr
from saher.pipeline import run_pipeline, VIOLATION_CLASS_NAMES
import numpy as np
import cv2


def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Saher Traffic Violation Detection")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="numpy", label="Upload Image")
                run_button = gr.Button("Run Detection")
            with gr.Column():
                gr.Markdown("### Violations Detected")
                output_gallery = gr.Gallery(
                    label="Violation Images with Bounding Boxes", columns=2
                )
                gr.Markdown("### License Plates")
                plate_gallery = gr.Gallery(label="License Plates", columns=3)

        def process_image(image):
            if image is None:
                return [], []

            violation_results, plate_images, ocr_results = run_pipeline([image])

            if not violation_results or not plate_images:
                return [], []

            violation_images_with_boxes = []
            plate_images_with_text = []

            for i in range(len(ocr_results)):
                if not violation_results[i].boxes:
                    continue

                # Get the annotated image with bounding boxes from YOLO
                annotated_image = violation_results[i].plot()
                # Convert from BGR to RGB
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                violation_images_with_boxes.append(
                    (annotated_image, f"Violation: {ocr_results[i]}")
                )

                # Add license plate image with OCR text
                plate_images_with_text.append(
                    (plate_images[i], f"Plate: {ocr_results[i]}")
                )

            return violation_images_with_boxes, plate_images_with_text

        run_button.click(
            fn=process_image,
            inputs=input_image,
            outputs=[output_gallery, plate_gallery],
        )
    demo.launch()


if __name__ == "__main__":
    main()
