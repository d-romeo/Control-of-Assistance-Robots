import torch
import numpy as np
import cv2



class YoloDetector:
    def __init__(self, yolo_path:str, weights_path:str, class_names:list, device:str = "cpu"):
        self.model = torch.hub.load(yolo_path, 'custom', path=weights_path, force_reload=True, source='local', device=device)
        self.class_names = class_names
    
    def inference(self, rgb_image:np.ndarray, res_x:int, res_y:int, min_conf:float=0.30):
        prediction = self.model([rgb_image], size=max([res_y,res_x]))
        results = []
        for detection in prediction.xyxy:
            for d in detection:
                x_min, y_min, x_max, y_max, conf, label_idx = d.numpy().tolist()
                if conf >= min_conf:
                    x_mid = (x_min + x_max)/2
                    y_mid = (y_min + y_max)/2
                    res = {"class_name": self.class_names[int(label_idx)],
                        "x_range": (int(x_min), int(x_max)),
                        "y_range": (int(y_min), int(y_max)),
                        "confidence": conf,
                        "midpoint": (int(x_mid), int(y_mid)),
                        "world_coord": None}
                    results.append(res)
        return results
    
    def save_results(self, rgb_image:np.ndarray, results:list, save_path:str, hide_conf:bool=True):
        
        final_image = rgb_image
        for res in results:
            label = res["class_name"] if hide_conf else f'{res["class_name"]} {res["confidence"]:.2f}'

            final_image = cv2.rectangle(final_image, (res["x_range"][0], res["y_range"][0]), 
                                  (res["x_range"][1], res["y_range"][1]), 
                                  [0,0,255], 5) 
            
            final_image = cv2.putText(final_image, label, (res["x_range"][0]-5, res["y_range"][0]-5), cv2.FONT_HERSHEY_SIMPLEX ,  
                   0.75, [0,0,255], 1, cv2.LINE_AA) 
            
            cv2.circle(final_image, res["midpoint"], 5, [255,0,0], -1)
        cv2.imwrite(save_path, final_image) 