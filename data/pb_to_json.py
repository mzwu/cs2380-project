#!/usr/bin/env python3
"""
Script to convert a .pb file to JSON format matching the structure in approx_max_group_pb.py.
"""

import csv
import json
from collections import defaultdict
from typing import Dict, List, Set

INPUT_FILE = "data/poland_warszawa_2021_.pb"
OUTPUT_FILE = "data/poland_warszawa_2021_.json"


def parse_pb_file(file_path: str) -> tuple[Dict[str, int], Dict[str, Set[str]], List[Set[str]]]:
    """
    Parse the .pb file and extract:
    - project_costs: dictionary mapping project_id (string) to cost (int)
    - groups: dictionary mapping category (string) to set of project_ids (strings)
    - approvals: list of sets, where each set contains project IDs approved by a voter
    """
    project_costs = {}
    groups = defaultdict(set)
    approvals = []
    
    in_projects_section = False
    in_votes_section = False
    header_line = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n\r')
            
            if line == "PROJECTS":
                in_projects_section = True
                continue
            
            if line == "VOTES":
                in_votes_section = True
                in_projects_section = False
                continue
            
            if in_projects_section:
                if line.startswith("project_id"):
                    header_line = line
                    continue
                
                if header_line and line:
                    reader = csv.reader([line], delimiter=';')
                    parts = next(reader)
                    
                    if parts and parts[0].isdigit():
                        project_id = parts[0]
                        cost = int(parts[1])
                        category = parts[4] if len(parts) > 4 else ""
                        
                        # Add to project_costs
                        project_costs[project_id] = cost
                        
                        # Add to groups by category
                        if category:
                            groups[category].add(project_id)
                        else:
                            groups["uncategorized"].add(project_id)
            
            elif in_votes_section:
                if line.startswith("voter_id"):
                    continue
                
                if line:
                    reader = csv.reader([line], delimiter=';')
                    parts = next(reader)
                    
                    if len(parts) >= 2:
                        vote_str = parts[1]
                        if vote_str:
                            # Parse comma-separated project IDs
                            project_ids = {pid.strip() for pid in vote_str.split(',') if pid.strip()}
                            approvals.append(project_ids)
    
    return project_costs, dict(groups), approvals


def convert_sets_to_lists(data):
    """Recursively convert sets to lists for JSON serialization."""
    if isinstance(data, dict):
        return {key: convert_sets_to_lists(value) for key, value in data.items()}
    elif isinstance(data, set):
        return sorted(list(data))  # Sort for consistent output
    elif isinstance(data, list):
        return [convert_sets_to_lists(item) for item in data]
    else:
        return data


def main():
    """Main function to convert PB file to JSON."""
    print(f"Reading file: {INPUT_FILE}")
    
    # Parse the file
    project_costs, groups, approvals = parse_pb_file(INPUT_FILE)
    
    print(f"Found {len(project_costs)} projects")
    print(f"Found {len(groups)} categories")
    print(f"Found {len(approvals)} voters")
    
    # Create the output structure
    output_data = {
        "project_costs": project_costs,
        "approvals": approvals,
        "groups": groups
    }
    
    # Convert sets to lists for JSON
    json_data = convert_sets_to_lists(output_data)
    
    # Write to JSON file
    print(f"\nWriting JSON file: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print("Done!")


if __name__ == "__main__":
    main()

