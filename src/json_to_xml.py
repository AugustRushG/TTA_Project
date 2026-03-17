import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def build_xml(parent: ET.Element, key: str, value: Any) -> None:
    if isinstance(value, dict):
        node = ET.SubElement(parent, key)
        for child_key, child_value in value.items():
            build_xml(node, str(child_key), child_value)
    elif isinstance(value, list):
        for item in value:
            build_xml(parent, key, item)
    else:
        node = ET.SubElement(parent, key)
        node.text = "" if value is None else str(value)


def data_to_xml_tree(data: Any) -> ET.ElementTree:
    if isinstance(data, dict) and len(data) == 1:
        root_key, root_value = next(iter(data.items()))
        root = ET.Element(str(root_key))
        if isinstance(root_value, dict):
            for key, value in root_value.items():
                build_xml(root, str(key), value)
        else:
            build_xml(root, "value", root_value)
    else:
        root = ET.Element("root")
        if isinstance(data, dict):
            for key, value in data.items():
                build_xml(root, str(key), value)
        elif isinstance(data, list):
            for item in data:
                build_xml(root, "item", item)
        else:
            build_xml(root, "value", data)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    return tree


def write_xml_data(data: Any, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    tree = data_to_xml_tree(data)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def json_to_xml(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    write_xml_data(data, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a JSON file to XML.")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", help="Path to output XML file (default: same name with .xml)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_suffix(".xml")
    json_to_xml(input_path, output_path)
    print(f"Converted {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
