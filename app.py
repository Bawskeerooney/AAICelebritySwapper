import time
import edgeiq
import numpy as np
import cv2
"""
Simultaneously use object detection to detect human faces and classification to classify
the detected faces in terms of age groups, and output results to
shared output stream.

To change the computer vision models, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_model.html

To change the engine and accelerator, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_engine_and_accelerator.html
"""
            

def main():
    # first make a detector to detect facial objects
    facial_detector = edgeiq.ObjectDetection(
            "alwaysai/res10_300x300_ssd_iter_140000")
    facial_detector.load(engine=edgeiq.Engine.DNN)

    # descriptions printed to console
    print("Engine: {}".format(facial_detector.engine))
    print("Accelerator: {}\n".format(facial_detector.accelerator))
    print("Model:\n{}\n".format(facial_detector.model_id))

    image_path = r"C:\Users\Aidan Rooney\OneDrive\Desktop\AlwaysAIPrograms\celebrity-swapper\images\kanyeWestPng.png"#Celebrity-Swapper retrieves path for images
    print("Images:\n{}\n".format(image_path))#Celebrity-Swapper Displays Video filenames
    fps = edgeiq.FPS()


    def bbox_cropper(image, bbox):
        h,w = image.shape[:2]
        bbox.start_x = max(0, bbox.start_x)
        bbox.start_y = max(0, bbox.start_y)
        bbox.end_x = min(w, bbox.end_x)
        bbox.end_y = min(h, bbox.end_y)

        return bbox



    def swap_faces(cbox, nbox, path, image1, image2):
        """[summary]

        Args:
            obox (BoundingBox): Will change in this function to the celebrity box
            nbox (BoundingBox): The original bounding box of the first person in the camera
            image1 (numpy array): The original frame
            image2 (numpy array): The new frame

        Returns:
            numpy array: The new numpy array with the new face on the old face
        """
        #Crop bounding box to size of the image
        celebrity_frame = cv2.imread(path)#Celebrity-Swapper image loaded of celebrity by retrieving its path    
        celebrity_results = facial_detector.detect_objects(celebrity_frame, confidence_level = 0.4)#Celebrity-Swapper will return the ObjectDetectionResults Object for all potential Objects in the frame
        cbox = celebrity_results.predictions[0].box #Celebrity-Swapper Gives us the bounding box for our celebrity
        print("bounding box:",cbox)
        print("shape of the image :",image1.shape)
        cbox = bbox_cropper(celebrity_frame, cbox)

        print("cbox after cropping :",cbox)
        face = edgeiq.resize(edgeiq.cutout_image(celebrity_frame, cbox), nbox.width , nbox.height) #Needs to be a numpy array of kanye for the image
        image2[nbox.start_y:nbox.start_y + face.shape[0], nbox.start_x:nbox.start_x + face.shape[1]] = face
        return image2              

    try:    
        # loop detection
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, edgeiq.Streamer() as streamer:
            time.sleep(2.0)
            fps.start()
            while True:
                frame = video_stream.read()
                print(frame.ndim)
                # detect human faces
                results = facial_detector.detect_objects(
                        frame, confidence_level=.4)
                


                # append each predication to the text output
                pred = results.predictions
                frame2 = frame.copy()
                for i in range(0, len(pred)):
                    if len(pred) != 1:   
                        if i < len(pred) - 1:
                            obox = pred[i].box
                            nbox = pred[i + 1].box
                        else:
                            obox = pred[i].box
                            nbox = pred[0].box
                        frame2 = swap_faces(obox, nbox, frame, frame2)

                if len(pred) > 0:
                    # swap the last face with the first face
                    obox = pred[len(pred) - 1].box
                    nbox = pred[0].box
                    frame2 = swap_faces(obox, nbox, image_path, frame, frame2)

                # send the image frame and the predictions to the output stream
                frame = edgeiq.markup_image(
                        frame, results.predictions, show_labels=False)
                display_frame = np.concatenate((frame, frame2))
                streamer.send_data(display_frame, "")

                fps.update()
                streamer.wait()

                if streamer.check_exit():
                    break

    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")
        exit()


if __name__ == "__main__":
    main()
