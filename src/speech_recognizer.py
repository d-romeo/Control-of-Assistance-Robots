import speech_recognition as sr
import time


class SpeechRecognizer:
    def __init__(self, object_names:list, capture_count_max:int=3) -> None:
        self.object_names = object_names.copy()
        self.recognizer = sr.Recognizer()
        self.capture_count_max = capture_count_max
    
    def request_object(self, coppelia_int_ui, coppelia_ui,id_sim_window) -> (bool, str):
        stop = False
        with sr.Microphone() as source:
            try:
                # Remove background noise
                self.recognizer.adjust_for_ambient_noise(source)
                # Get audio from microphone and use google services to transcribe it
                for i in range(0, self.capture_count_max):
                    print(f"[INFO] Capturing your voice in {self.capture_count_max - i}...")
                    coppelia_int_ui.set_label(coppelia_ui,f"[INFO] Capturing your voice in {self.capture_count_max - i}...",id_sim_window)
                    time.sleep(1)
                print(f"[INFO] Speak into the microphone.")
                coppelia_int_ui.set_label(coppelia_ui,f"[INFO] Speak into the microphone.",id_sim_window)
                audio_data = self.recognizer.listen(source)
                print(f"[INFO] Translating the audio into text...")
                coppelia_int_ui.set_label(coppelia_ui,f"[INFO] Translating the audio into text...",id_sim_window)
                transcribed_audio = self.recognizer.recognize_google(audio_data, language="it-IT")
                print(f"[INFO] Audio translated. You said \'{transcribed_audio}\'")
                coppelia_int_ui.set_label(coppelia_ui,f"[INFO] Audio translated. You said \'{transcribed_audio}\'",id_sim_window)
                transcribed_audio_list = transcribed_audio.split(" ")
                
                # Check if the transcribed audio contains a key command
                result, object_name, stop = self.check_command_validity(transcribed_audio_list)

                return result, object_name, stop

            except Exception as e:
                print(F"[ERROR] {e}")
    
    def check_command_validity(self, words_to_check:list) -> (bool, str, bool):        
        for i in range(len(words_to_check)):
            words_to_check[i] = words_to_check[i].upper()

        for name in self.object_names:
            if name.upper() in words_to_check:
                return True, name, False
        if "STOP" in words_to_check:
            return False, "NOT FOUND", True
        return False, "NOT FOUND", False
