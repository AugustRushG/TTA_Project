import xmltodict
import json

xml_file = "/home/s224705071/github/TTA_Project/src/25WPF JPN M11 F_Kim Gi Tae KOR v Kato JPN.xml"
json_file = "output.json"

with open(xml_file, "r", encoding="utf-16") as f:
    data_dict = xmltodict.parse(f.read())

with open(json_file, "w", encoding="utf-8") as f:
    json.dump(data_dict, f, indent=4, ensure_ascii=False)

print(f"Converted {xml_file} -> {json_file}")