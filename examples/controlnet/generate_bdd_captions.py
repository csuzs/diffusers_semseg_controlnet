import json



def create_bdd_captions(frame_metadata_path: str, caption_txt_output_folder: str):
    with open(frame_metadata_path) as f:
        mtdt_json = json.load(frame_metadata_path)
        for i in range(len(mtdt_json)):
            frame_mtdt = mtdt_json[i]
            frame_name = frame_mtdt['name'].split(".")[0] 
            attributes = frame_mtdt[i]['attributes']
            prompt = "Traffic scene. "
            if attributes["weather"] != "undefined":
                prompt += f"{attributes["weather"]} weather. "
            
            if attributes["timeofday"] != "undefined":
                prompt += f"{attributes["timeofday"]}. "
            
            if attributes["scene"] != "undefined":
                prompt += f"{attributes["scene"]}."
            
            with open(f"{frame_name}.txt") as cap_file:
                cap_file.write(prompt)
        
if __name__ == "__main__":
    create_bdd_captions(frame_metadata_path="/project_workspace/uic19759/bdd/bdd100k/labels/det_20/det_train.json",caption_txt_output_folder=".")