from speech_recognizer import SpeechRecognizer

translation_dict = {'COLTELLO': "KINFE", 'FORCHETTA': "FORK", 'TAZZA':'MUG', 'BICCHIERE':'CUP', 'CUCCHIAIO': 'SPOON', 'MEDICINA': 'MEDICINE'}

if __name__ == "__main__":
    sp_rec_object_names = ["COLTELLO", "FORCHETTA", "TAZZA", "CUCCHIAIO", "MEDICINA", "BICCHIERE"]
    sp_rec = SpeechRecognizer(object_names=sp_rec_object_names)

    result, nome_italiano = sp_rec.request_object()
    if result:
        english_name = translation_dict[nome_italiano]
        print(english_name)
    else:
        print("invalid command")


