#%%
import pathlib
import re
import argparse
import pandas as pd


def transform_data(input_data_path, output_data_path):
    """Transform input structure of:
        1:12:clinical_variable,23:32:upper_bound	temperature less than @NUMBER f

        to: temperature less than @NUMBER f, [
            {'start': 0, 'end': 11, 'label': 'clinical_variable'},
            {'start': 22, 'end': 31, 'label': 'upper_bound'}]"""


    input_data_path = pathlib.Path(input_data_path)
    print(input_data_path)
    output_data_path = pathlib.Path(output_data_path)

    entities = r"(\d{1,3}):(\d{1,3}):([a-z]*_?[a-z]*)"
    negated_text = r"\d{1,3}:\d{1,3}:[a-z]*_?[a-z]*,?\s?"

    with input_data_path.open() as file:
        ner_list = []
        for _, line in enumerate(file):
            matched_entities = re.findall(entities, line)
            
            matched_entities_formatted = []
            for matched_entity in matched_entities:
                matched_entity_0 = int(matched_entity[0])-1
                matched_entity_1 = int(matched_entity[1])-1
                matched_entity_2 = str(matched_entity[2])
                annotations = {
                    "start": matched_entity_0,
                    "end": matched_entity_1,
                    "label": matched_entity_2
                }
                matched_entities_formatted.append(annotations)

            matched_text = re.sub(negated_text, '', line).replace('\n', '')
            df_ner = pd.DataFrame({"x": matched_text, "y": [matched_entities_formatted]})
            ner_list.append(df_ner)

    df = pd.concat(ner_list)
    df.to_csv(output_data_path, index=False)


#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--input_data_path',
            help="Path to the input data."
    )

    parser.add_argument(
            '--output_data_path',
            help="Path to the output data."
    )

    args = parser.parse_args()
    transform_data(
        args.input_data_path,
        args.output_data_path)
# %%
