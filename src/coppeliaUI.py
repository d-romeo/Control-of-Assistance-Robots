from coppeliasim_zmqremoteapi_client  import RemoteAPIClient

class CoppeliaSimUI:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.simUI = self.client.require('simUI')
    
    def init_window(self):
        id_sim_window= {"changing_label": 1000,
                "main_image": 5007}
        xml = '<ui title="Robot Assistant" closeable="true" resizeable="true" activate="true" position="1550,500" placement ="absolute">'+ \
            '<label text="Welcome to Robot Assistant, please follow the instructions below."  wordwrap="true" />' + \
            '<text-browser id="1000" text = "[INFO] Wait! The robot is positioning itself"/>'+ \
            '<label text="System identified: "  wordwrap="true" />' + \
            '<image id="5007" scaled-contents = "true"/>'+ \
            '</ui>'
        ui = self.simUI.create(xml)
        return ui,id_sim_window
    
    def set_label(self,ui,text,id_sim_window,suppressEvents: bool = False): 
        self.simUI.setText(ui,id_sim_window['changing_label'],text)

    def set_img(self,sim,ui,id_sim_window,abs_path):
        img, res = sim.load_image(abs_path)
        self.simUI.setImageData(ui,id_sim_window['main_image'],img,res[0],res[1])    