'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
from openvino.inference_engine import IECore

class FaceDetectionModel:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device = 'CPU', extensions = None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.model_structure = self.model_name
        self.model_weights = self.model_name.split('.')[0]+'.bin'
        self.plugin = None
        self.network = None
        self.exec_net = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None
        self.output_shape = None

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.plugin = IECore()
        self.network = self.plugin.read_network(model = self.model_structure, weights = self.model_weights)
        supported_layers = self.plugin.query_network(network = self.network, device_name = self.device)
        unsupported_layers = [i for i in self.network.layers.keys() if i not in supported_layers]
        
        
        if len(unsupported_layers) != 0 and self.device == 'CPU':
            print("unsupported layers found:{}".format(unsupported_layers))
            if not self.extensions==None:
                print("Adding cpu_extension")
                self.plugin.add_extension(self.extensions, self.device)
                supported_layers = self.plugin.query_network(network = self.network, device_name=self.device)
                unsupported_layers = [i for i in self.network.layers.keys() if i not in supported_layers]
                if len(unsupported_layers) != 0:
                    print("After adding the extension still unsupported layers found")
                    exit(1)
                print("After adding the extension the issue is resolved")
            else:
                print("Give the path of cpu extension")
                exit(1)
                
        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device,num_requests=1)
        
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_names = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_names].shape
        
    def predict(self, image, prob_threshold):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        
        img_processed = self.preprocess_input(image.copy())
        outputs = self.exec_net.infer({self.input_name : img_processed})
        coordinates = self.preprocess_output(outputs, prob_threshold)
        if (len(coordinates) == 0):
            return 0, 0
        coordinates = coordinates[0] #take the first detected face
        h = image.shape[0]
        w = image.shape[1]
        coordinates = coordinates * np.array([w, h, w, h])
        coordinates = coordinates.astype(np.int32)
        
        face_cropped = image[coordinates[1] : coordinates[3], coordinates[0] : coordinates[2]]
        return face_cropped, coordinates


    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        resized_img = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        processed_img = np.transpose(np.expand_dims(resized_img, axis=0), (0, 3, 1, 2))
        return processed_img
            

    def preprocess_output(self, outputs, prob_threshold):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        coordinates = []
        outs = outputs[self.output_names][0][0]
        for out in outs:
            conf = out[2]
            if conf > prob_threshold:
                x_min = out[3]
                y_min = out[4]
                x_max = out[5]
                y_max = out[6]
                coordinates.append([x_min, y_min, x_max, y_max])
        return coordinates