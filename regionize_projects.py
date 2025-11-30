"""
Script to categorize projects in a .pb file into geographic regions.
Assigns projects to 4 named regions (Syrenka, Marysin, Anin, SP218) based on
project names and coordinate proximity, producing data structures for hierarchical_pb.py.
"""

import csv
import json
import math
from typing import Dict, List, Tuple, Set

# File paths
INPUT_FILE = "data/poland_warszawa_2026_marysin-wawerski-anin.pb"
OUTPUT_FILE = "data/region_assignments.json"

# Region definitions with keyword patterns
REGION_KEYWORDS = {
    "Syrenka": ["syrence", "syrenka", "syrenc"],
    "Marysin": ["marysin"],
    "Anin": ["anin"],
    "SP218": ["218", "kajki"],
}

# Known project assignments from plan (by name analysis)
KNOWN_ASSIGNMENTS = {
    # Syrenka projects (mention Syrence/Syrenka)
    "366": "Syrenka",   # Kino plenerowe na Syrence
    "479": "Syrenka",   # Potańcówki w na Syrence
    "880": "Syrenka",   # Hala namiotowa... (same coords as Syrence projects)
    "1571": "Syrenka",  # Trening tai chi na Syrence
    "370": "Syrenka",   # Zielona Wiata Przystankowa (near Syrence)
    
    # SP218 projects (Szkoła Podstawowa nr 218)
    "1098": "SP218",    # Budowa boiska... przy SP 218
    "1142": "SP218",    # Street Workout Park... przy SP 218
    "1179": "SP218",    # Zielona klasa... SP 218
    "1113": "SP218",    # Stoły do tenisa... SP 218
    
    # Marysin projects
    "958": "Marysin",   # Ławeczki w Marysinie (no coords, assigned per plan)
    "634": "Marysin",   # Książkomat... w Marysinie Wawerskim
    "2152": "Marysin",  # Tablice... w Marysinie i Aninie
    
    # Anin projects
    "209": "Anin",      # Książkomat dla Biblioteki Głównej w Aninie
    "68": "Anin",       # Otwarcie Skateparku w Aninie
}


def parse_pb_file(file_path: str) -> Tuple[List[Dict], int]:
    """Parse the .pb file and return project data and global budget."""
    projects = []
    global_budget = 0
    in_projects_section = False
    header_line = None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n\r')
            
            # Extract budget from metadata
            if line.startswith("budget;"):
                global_budget = int(line.split(";")[1])
                continue
                
            if line == "PROJECTS":
                in_projects_section = True
                continue
            if line == "VOTES":
                break
            if not in_projects_section:
                continue
                
            if line.startswith("project_id"):
                header_line = line
            elif line and header_line:
                reader = csv.reader([line], delimiter=';')
                parts = next(reader)

                if parts and parts[0].isdigit():
                    while len(parts) < 9:
                        parts.append('')

                    project = {
                        'project_id': parts[0],
                        'cost': int(parts[1]) if parts[1] else 0,
                        'votes': int(parts[2]) if parts[2] else 0,
                        'name': parts[3],
                        'category': parts[4],
                        'selected': parts[5] if len(parts) > 5 else '',
                        'latitude': float(parts[6]) if parts[6] else None,
                        'longitude': float(parts[7]) if parts[7] else None,
                    }
                    projects.append(project)

    return projects, global_budget


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points in meters."""
    R = 6371000  # Earth's radius in meters
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def assign_by_keywords(project: Dict) -> str | None:
    """Try to assign a project to a region based on name keywords."""
    name_lower = project['name'].lower()
    
    for region, keywords in REGION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in name_lower:
                return region
    return None


def compute_region_centroids(projects: List[Dict], assignments: Dict[str, str]) -> Dict[str, Tuple[float, float]]:
    """Compute the centroid (average lat/lon) for each region."""
    region_coords = {region: [] for region in REGION_KEYWORDS.keys()}
    
    for project in projects:
        pid = project['project_id']
        if pid in assignments and project['latitude'] is not None:
            region = assignments[pid]
            region_coords[region].append((project['latitude'], project['longitude']))
    
    centroids = {}
    for region, coords in region_coords.items():
        if coords:
            avg_lat = sum(c[0] for c in coords) / len(coords)
            avg_lon = sum(c[1] for c in coords) / len(coords)
            centroids[region] = (avg_lat, avg_lon)
    
    return centroids


def assign_by_proximity(project: Dict, centroids: Dict[str, Tuple[float, float]]) -> str:
    """Assign a project to the nearest region centroid."""
    if project['latitude'] is None or project['longitude'] is None:
        # Default to Marysin for projects without coordinates (as per plan)
        return "Marysin"
    
    min_dist = float('inf')
    nearest_region = "Marysin"  # Default fallback
    
    for region, (lat, lon) in centroids.items():
        dist = haversine_distance(project['latitude'], project['longitude'], lat, lon)
        if dist < min_dist:
            min_dist = dist
            nearest_region = region
    
    return nearest_region


def regionize_projects(projects: List[Dict]) -> Dict[str, str]:
    """Assign all projects to regions."""
    assignments = {}
    
    # First pass: Use known assignments from plan
    for pid, region in KNOWN_ASSIGNMENTS.items():
        assignments[pid] = region
    
    # Second pass: Try keyword matching for remaining projects
    for project in projects:
        pid = project['project_id']
        if pid not in assignments:
            region = assign_by_keywords(project)
            if region:
                assignments[pid] = region
    
    # Compute centroids from assigned projects
    centroids = compute_region_centroids(projects, assignments)
    print(f"Region centroids: {centroids}")
    
    # Third pass: Assign remaining by proximity
    for project in projects:
        pid = project['project_id']
        if pid not in assignments:
            region = assign_by_proximity(project, centroids)
            assignments[pid] = region
            print(f"  Assigned project {pid} ({project['name'][:40]}...) to {region} by proximity")
    
    return assignments


def generate_hierarchical_structures(
    projects: List[Dict], 
    assignments: Dict[str, str],
    global_budget: int
) -> Tuple[List[List[str]], Dict[str, List[str]], Dict[str, Set[str]], Dict[str, int]]:
    """Generate the data structures required by hierarchical_pb.py."""
    
    regions = list(REGION_KEYWORDS.keys())
    
    # Layers: top-level "All", then the 4 regions
    layers = [["All"], regions]
    
    # Hierarchy: All -> 4 regions, each region is a leaf
    hierarchy = {"All": regions}
    for region in regions:
        hierarchy[region] = []
    
    # Groups: mapping of group name to set of project IDs
    all_project_ids = set()
    groups = {region: set() for region in regions}
    
    for project in projects:
        pid = project['project_id']
        all_project_ids.add(pid)
        if pid in assignments:
            groups[assignments[pid]].add(pid)
    
    groups["All"] = all_project_ids
    
    # Group budgets: proportional to sum of project costs in each region
    region_costs = {region: 0 for region in regions}
    for project in projects:
        pid = project['project_id']
        if pid in assignments:
            region_costs[assignments[pid]] += project['cost']
    
    total_cost = sum(region_costs.values())
    group_budgets = {"All": global_budget}
    
    for region in regions:
        # Proportional budget based on project costs, with some buffer
        if total_cost > 0:
            proportion = region_costs[region] / total_cost
            group_budgets[region] = int(global_budget * proportion * 1.2)  # 20% buffer
        else:
            group_budgets[region] = global_budget // len(regions)
    
    return layers, hierarchy, groups, group_budgets


def main():
    """Main function to regionize projects."""
    print(f"Reading file: {INPUT_FILE}")
    projects, global_budget = parse_pb_file(INPUT_FILE)
    print(f"Found {len(projects)} projects, global budget: {global_budget}")
    
    print("\nAssigning projects to regions...")
    assignments = regionize_projects(projects)
    
    # Count projects per region
    region_counts = {}
    for region in assignments.values():
        region_counts[region] = region_counts.get(region, 0) + 1
    print(f"\nProjects per region: {region_counts}")
    
    print("\nGenerating hierarchical structures...")
    layers, hierarchy, groups, group_budgets = generate_hierarchical_structures(
        projects, assignments, global_budget
    )
    
    # Prepare output data
    output = {
        "assignments": assignments,
        "layers": layers,
        "hierarchy": hierarchy,
        "groups": {k: list(v) for k, v in groups.items()},  # Convert sets to lists for JSON
        "group_budgets": group_budgets,
        "project_costs": {p['project_id']: p['cost'] for p in projects},
    }
    
    # Save to JSON
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Print structures for hierarchical_pb.py
    print("\n" + "=" * 60)
    print("STRUCTURES FOR hierarchical_pb.py:")
    print("=" * 60)
    
    print(f"\nlayers = {layers}")
    print(f"\nhierarchy = {hierarchy}")
    print(f"\ngroups = {{")
    for group, pids in groups.items():
        print(f"    '{group}': {pids},")
    print("}")
    print(f"\ngroup_budgets = {group_budgets}")
    
    # Print detailed assignments
    print("\n" + "=" * 60)
    print("DETAILED ASSIGNMENTS:")
    print("=" * 60)
    for region in REGION_KEYWORDS.keys():
        print(f"\n{region}:")
        for project in projects:
            if assignments.get(project['project_id']) == region:
                coords = f"({project['latitude']:.4f}, {project['longitude']:.4f})" if project['latitude'] else "(no coords)"
                print(f"  {project['project_id']}: {project['name'][:50]}... {coords}")


if __name__ == "__main__":
    main()

