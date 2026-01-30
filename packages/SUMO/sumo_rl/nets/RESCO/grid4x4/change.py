#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import xml.etree.ElementTree as ET
import random
import numpy as np
from typing import List, Tuple


def parse_xml_and_extract_vehicles(xml_file: str, min_depart: float = 1500.0, max_depart: float = 3000.0) -> List[Tuple[str, str, str]]:

    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    vehicles = []
    
    for vehicle in root.findall('vehicle'):
        vehicle_id = vehicle.get('id')
        depart_attr = vehicle.get('depart')
        
        if depart_attr is None:
            continue
            
        depart_time = float(depart_attr)
        
        if min_depart <= depart_time <= max_depart:
            route = vehicle.find('route')
            if route is not None:
                route_edges = route.get('edges')
                if route_edges is not None:
                    vehicles.append((vehicle_id, str(depart_time), route_edges))
    
    return vehicles


def generate_uniform_random_times(num_vehicles: int, min_time: float = 1500.0, max_time: float = 3000.0) -> List[float]:

    times = np.random.uniform(min_time, max_time, num_vehicles)
    times.sort()
    return times.tolist()


def generate_extended_random_times(num_vehicles: int, min_time: float = 0.0, max_time: float = 15000.0) -> List[float]:

    times = np.random.uniform(min_time, max_time, num_vehicles)
    times.sort()
    return times.tolist()


def create_new_vehicle_elements(vehicles: List[Tuple[str, str, str]], 
                              new_times: List[float], 
                              base_id: int = 10000) -> List[ET.Element]:

    new_vehicles = []
    
    for i, ((original_id, original_depart, route_edges), new_time) in enumerate(zip(vehicles, new_times)):

        vehicle_elem = ET.Element('vehicle')
        new_id = f"{original_id}_copy_{i+1}"
        vehicle_elem.set('id', new_id)
        vehicle_elem.set('depart', f"{new_time:.2f}")
        
        route_elem = ET.SubElement(vehicle_elem, 'route')
        route_elem.set('edges', route_edges)
        
        new_vehicles.append(vehicle_elem)
    
    return new_vehicles


def process_xml_file(input_file: str, output_file: str, 
                    min_depart: float = 1500.0, max_depart: float = 3000.0,
                    multiplier: int = 10, base_id: int = 10000):

    print(f"is dealing with {input_file}")
    
    vehicles = parse_xml_and_extract_vehicles(input_file, min_depart, max_depart)
    print(f"find {len(vehicles)} cars in {min_depart}-{max_depart} ")
    
    if not vehicles:
        print("no vehicles found")
        return
    
    all_vehicles = vehicles * multiplier
    print(f" {multiplier} , {len(all_vehicles)} ")
    
    new_times = generate_extended_random_times(len(all_vehicles), 0.0, 15000.0)
    print(f" {len(new_times)} ")
    
    new_vehicle_elements = create_new_vehicle_elements(all_vehicles, new_times, base_id)
    print(f" {len(new_vehicle_elements)} ")
    
    tree = ET.parse(input_file)
    root = tree.getroot()
    

    for vehicle_elem in new_vehicle_elements:
        root.append(vehicle_elem)
    
    xml_str = ET.tostring(root, encoding='unicode')
    
    import re
    xml_str = re.sub(r'><', '>\n    <', xml_str)
    xml_str = re.sub(r'<vehicle', '    <vehicle', xml_str)
    xml_str = re.sub(r'<route', '        <route', xml_str)
    xml_str = re.sub(r'</route>', '        </route>', xml_str)
    xml_str = re.sub(r'</vehicle>', '    </vehicle>', xml_str)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(xml_str)
    print(f"{output_file}")
    
    print(f"\n")
    print(f" {len(vehicles)}")
    print(f" {multiplier}")
    print(f"{len(new_vehicle_elements)}")
    print(f"{min_depart} - {max_depart}")
    print(f" {base_id} - {base_id + len(new_vehicle_elements) - 1}")


def main():
    input_file = "grid4x4_1.rou.xml"
    output_file = "grid4x4_1.rou.xml"  
    
    random.seed(42)
    np.random.seed(42)
    
    try:
        process_xml_file(
            input_file=input_file,
            output_file=output_file,
            min_depart=1500.0,
            max_depart=3000.0,
            multiplier=10,
            base_id=10000
        )
        print("\ndone！")
        
    except Exception as e:
        print(f" {e}")


if __name__ == "__main__":
    main()
