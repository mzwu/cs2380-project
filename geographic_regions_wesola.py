"""
Script to categorize projects in poland_warszawa_2023_wesola.pb (Wesoła district)
into geographic regions based on project names and coordinate proximity.

Wesoła district neighborhoods:
- Wola Grzybowska (northern part, near train station)
- Stara Miłosna (central area)
- Groszówka (southern area)
"""

import csv
import json
import math
from typing import Dict, List, Tuple, Set

# File paths
INPUT_FILE = "data/poland_warszawa_2023_wesola.pb"
OUTPUT_FILE = "data/region_assignments_wesola.json"

# Region definitions with keyword patterns for Wesoła neighborhoods
REGION_KEYWORDS = {
    "Wola Grzybowska": ["wola grzybowska", "grzybowsk"],
    "Stara Miłosna": ["miłosna", "milosna", "rolkostrad"],
    "Groszówka": ["groszówk", "groszowk", "granicz"],
}

# Known project assignments based on name/location analysis
KNOWN_ASSIGNMENTS = {
    # Wola Grzybowska (northern area, near PKP station)
    # System monitoringu parkingu przy stacji PKP Wola Grzybowska
    "726": "Wola Grzybowska",

    # Stara Miłosna (central area with Rolkostrada)
    "548": "Stara Miłosna",     # Rolki i nartorolki na Rolkostradzie
    "1775": "Stara Miłosna",    # Przyrodnicze Skarby Rolkostrady
    "1042": "Stara Miłosna",    # Książkomat (near Rolkostrada coords)

    # Groszówka (southern area, ul. Graniczna)
    "1778": "Groszówka",        # Zielona zasłona przy ulicy Granicznej
}


def parse_pb_file(file_path: str) -> Tuple[List[Dict], int]:
    """Parse the .pb file and return project data and global budget."""
    projects = []
    global_budget = 0
    in_projects_section = False
    header_line = None
    header_columns = None

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
                header_columns = line.split(';')
                print(f"Header columns: {header_columns}")
            elif line and header_line:
                reader = csv.reader([line], delimiter=';')
                parts = next(reader)

                if parts and parts[0].isdigit():
                    while len(parts) < 9:
                        parts.append('')

                    # Find latitude/longitude columns
                    lat_idx = header_columns.index(
                        'latitude') if 'latitude' in header_columns else 7
                    lon_idx = header_columns.index(
                        'longitude') if 'longitude' in header_columns else 8

                    project = {
                        'project_id': parts[0],
                        'cost': int(parts[1]) if parts[1] else 0,
                        'votes': int(parts[2]) if parts[2] else 0,
                        'name': parts[3],
                        'category': parts[4] if len(parts) > 4 else '',
                        'selected': parts[6] if len(parts) > 6 else '',
                        'latitude': float(parts[lat_idx]) if parts[lat_idx] else None,
                        'longitude': float(parts[lon_idx]) if parts[lon_idx] else None,
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
            if region in region_coords:
                region_coords[region].append(
                    (project['latitude'], project['longitude']))

    centroids = {}
    for region, coords in region_coords.items():
        if coords:
            avg_lat = sum(c[0] for c in coords) / len(coords)
            avg_lon = sum(c[1] for c in coords) / len(coords)
            centroids[region] = (avg_lat, avg_lon)

    # Default centroids for Wesoła neighborhoods based on geography
    # Wesoła is roughly: lat 52.21-52.26, lon 21.20-21.26
    default_centroids = {
        "Wola Grzybowska": (52.252, 21.252),  # Northern, near PKP station
        "Stara Miłosna": (52.235, 21.220),     # Central area
        "Groszówka": (52.215, 21.225),         # Southern area
    }

    for region in REGION_KEYWORDS.keys():
        if region not in centroids:
            centroids[region] = default_centroids[region]
            print(
                f"  Using default centroid for {region}: {default_centroids[region]}")

    return centroids


def assign_by_proximity(project: Dict, centroids: Dict[str, Tuple[float, float]]) -> str:
    """Assign a project to the nearest region centroid."""
    if project['latitude'] is None or project['longitude'] is None:
        # Default to Stara Miłosna (central) for projects without coordinates
        return "Stara Miłosna"

    min_dist = float('inf')
    nearest_region = "Stara Miłosna"  # Default fallback

    for region, (lat, lon) in centroids.items():
        dist = haversine_distance(
            project['latitude'], project['longitude'], lat, lon)
        if dist < min_dist:
            min_dist = dist
            nearest_region = region

    return nearest_region


def regionize_projects(projects: List[Dict]) -> Dict[str, str]:
    """Assign all projects to regions."""
    assignments = {}

    # First pass: Use known assignments
    for pid, region in KNOWN_ASSIGNMENTS.items():
        assignments[pid] = region

    # Second pass: Try keyword matching for remaining projects
    for project in projects:
        pid = project['project_id']
        if pid not in assignments:
            region = assign_by_keywords(project)
            if region:
                assignments[pid] = region
                print(
                    f"  Assigned {pid} ({project['name'][:40]}...) to {region} by keyword")

    # Compute centroids from assigned projects
    centroids = compute_region_centroids(projects, assignments)
    print(f"\nRegion centroids: {centroids}")

    # Third pass: Assign remaining by proximity
    for project in projects:
        pid = project['project_id']
        if pid not in assignments:
            region = assign_by_proximity(project, centroids)
            assignments[pid] = region
            print(
                f"  Assigned {pid} ({project['name'][:40]}...) to {region} by proximity")

    return assignments


def generate_hierarchical_structures(
    projects: List[Dict],
    assignments: Dict[str, str],
    global_budget: int
) -> Tuple[List[List[str]], Dict[str, List[str]], Dict[str, Set[str]], Dict[str, int]]:
    """Generate the data structures required by hierarchical_pb.py."""

    regions = list(REGION_KEYWORDS.keys())

    # Layers: top-level "All", then the 3 regions
    layers = [["All"], regions]

    # Hierarchy: All -> regions, each region is a leaf
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
            group_budgets[region] = int(
                global_budget * proportion * 1.2)  # 20% buffer
        else:
            group_budgets[region] = global_budget // len(regions)

    return layers, hierarchy, groups, group_budgets


def main():
    """Main function to regionize projects."""
    print(f"Reading file: {INPUT_FILE}")
    projects, global_budget = parse_pb_file(INPUT_FILE)
    print(f"Found {len(projects)} projects, global budget: {global_budget:,}")

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
        "groups": {k: list(v) for k, v in groups.items()},
        "group_budgets": group_budgets,
        "project_costs": {p['project_id']: p['cost'] for p in projects},
    }

    # Save to JSON
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nlayers = {layers}")
    print(f"\nhierarchy = {hierarchy}")
    print(f"\ngroup_budgets = {group_budgets}")

    # Print detailed assignments
    print("\n" + "=" * 60)
    print("DETAILED ASSIGNMENTS:")
    print("=" * 60)
    for region in REGION_KEYWORDS.keys():
        region_projects = [p for p in projects if assignments.get(
            p['project_id']) == region]
        total_cost = sum(p['cost'] for p in region_projects)
        print(
            f"\n{region} ({len(region_projects)} projects, total cost: {total_cost:,}):")
        for project in region_projects:
            coords = f"({project['latitude']:.4f}, {project['longitude']:.4f})" if project['latitude'] else "(no coords)"
            print(
                f"  {project['project_id']}: {project['name'][:50]}... {coords}")


if __name__ == "__main__":
    main()
