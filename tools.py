from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image

#Since in our workflow, the langchain model requires two tools, this python file provides the description of those tools, namely
    #Image Caption Tool
    #Object Detection Tool
#

class ImageCaptionTool(BaseTool):
    name = "Image Caption Generator"
    description = "This tool is used to describe an image given its path." \
                    "It will return a short caption of the image."

    def _run(self, image_path):
        """Generates a short caption for the provided image
    
        Arguments: image_path(str): Path of input file
        Return: str: String which is the caption
        
        Uses transformers library
        """

        image = Image.open(image_path).convert('RGB')

        model_name = "Salesforce/blip-image-captioning-large"
        device = "cpu"

        processor = BlipProcessor.from_pretrained(model_name)

        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

        # Converting our image into another form 
        inputs = processor(image, return_tensors='pt').to(device)

        output = model.generate(**inputs, max_new_tokens=40)

        caption = processor.decode(output[0], skip_special_tokens = True)

        return caption

    def _arun(self, query: str):
        raise NotImplementedError("This Tool does not support async")
    
class ObjectDetectionTool(BaseTool):
    name = "Object Detection Tool"
    description = "This tool is used to detect object in an image given its path." \
                    "It will return a list of all detected objects in the following format:"\
                    "[x1,y1,x2,y2] class_name confidence_score."

    def _run(self, image_path):
        """
        Detects objects in the provided image.

        Args: image_path (str): The path to the image file.

        Returns: str: A string with all the detected objects. Each object as '[x1, x2, y1, y2, class_name, confindence_score]'.
        """

        image = Image.open(image_path).convert('RGB')

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detections = ""
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            detections += ' {}'.format(model.config.id2label[int(label)])
            detections += ' {}\n'.format(float(score))

        return detections
    
    def _arun(self, query: str):
        raise NotImplementedError("This Tool does not support async")
