"""
Script to categorize projects in a .pb file using OpenAI API.
Reads project descriptions, sends them to OpenAI for categorization,
and updates the category field with the single most relevant category.
"""

import csv
import shutil
from typing import List, Dict, Tuple
from openai import OpenAI

OPENAI_API_KEY = "CHANGE_ME"

# File paths (UPDATE TO YOUR FILE PATHS)
INPUT_FILE = "data/poland_warszawa_2021_.pb"
BACKUP_FILE = "data/poland_warszawa_2021_.pb.backup"


def extract_categories_from_file(file_path: str) -> set:
    """Extract all unique categories from the projects in the file."""
    categories = set()
    in_projects_section = False

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == "PROJECTS":
                in_projects_section = True
                continue
            if line == "VOTES":
                break
            if in_projects_section and line and not line.startswith("project_id"):
                reader = csv.reader([line], delimiter=';')
                parts = next(reader)
                if parts and parts[0].isdigit() and len(parts) >= 5:
                    category_field = parts[4]
                    if category_field:
                        for cat in category_field.split(','):
                            cat = cat.strip()
                            if cat:
                                categories.add(cat)

    return categories


def parse_pb_file(file_path: str) -> Tuple[List[str], List[Dict], List[str]]:
    """Parse the .pb file and return META lines, project data, and remaining lines (VOTES section)."""
    meta_lines = []
    projects = []
    remaining_lines = []
    in_projects_section = False
    header_line = None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n\r')
            if line == "PROJECTS":
                in_projects_section = True
                meta_lines.append(line)
                continue
            if line == "VOTES":
                remaining_lines.append(line)
                for remaining_line in f:
                    remaining_lines.append(remaining_line.rstrip('\n\r'))
                break
            if not in_projects_section:
                meta_lines.append(line)
            else:
                if line.startswith("project_id"):
                    header_line = line
                    meta_lines.append(line)  # Header line
                elif line and header_line:
                    reader = csv.reader([line], delimiter=';')
                    parts = next(reader)

                    if parts and parts[0].isdigit():
                        while len(parts) < 9:
                            parts.append('')

                        project = {
                            'line': line,
                            'project_id': parts[0],
                            'cost': parts[1],
                            'votes': parts[2],
                            'name': parts[3],
                            'category': parts[4],
                            'target': parts[5],
                            'selected': parts[6],
                            'latitude': parts[7] if len(parts) > 7 else '',
                            'longitude': parts[8] if len(parts) > 8 else ''
                        }
                        projects.append(project)

    return meta_lines, projects, remaining_lines


def get_category_from_openai(client: OpenAI, description: str, valid_categories: List[str]) -> str:
    """Send project description to OpenAI and get the single most relevant category."""
    categories_list = ', '.join(sorted(valid_categories))

    prompt = f"""You are categorizing municipal projects in Poland. Given the Polish project description below, select the SINGLE most relevant category from this list:

{categories_list}

Project description: {description}

Return ONLY the category name from the list above, nothing else. Do not include any explanation or additional text."""

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini-2025-08-07",
            messages=[
                {"role": "system", "content": "You are an assistant that categorizes municipal projects in Poland. Always return only the most relevant category name, nothing else."},
                {"role": "user", "content": prompt}
            ],
        )

        category = response.choices[0].message.content.strip()

        category_lower = category.lower()
        for valid_cat in valid_categories:
            if valid_cat.lower() == category_lower:
                return valid_cat

        print(
            f"Warning: OpenAI returned '{category}', not in valid categories. Using first valid category as fallback.")
        return sorted(valid_categories)[0]

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return sorted(valid_categories)[0]


def write_pb_file(file_path: str, meta_lines: List[str], projects: List[Dict], remaining_lines: List[str]):
    """Write the updated data back to the .pb file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in meta_lines:
            f.write(line + '\n')

        writer = csv.writer(f, delimiter=';', lineterminator='\n')
        for project in projects:
            row = [
                project['project_id'],
                project['cost'],
                project['votes'],
                project['name'],
                project['category'],
                project['target'],
                project['selected'],
                project['latitude'],
                project['longitude']
            ]
            writer.writerow(row)

        for line in remaining_lines:
            f.write(line + '\n')


def main():
    """Main function to categorize all projects."""
    print(f"Reading file: {INPUT_FILE}")

    print(f"Creating backup: {BACKUP_FILE}")
    shutil.copy2(INPUT_FILE, BACKUP_FILE)

    print("Extracting valid categories from file...")
    valid_categories = extract_categories_from_file(INPUT_FILE)
    print(
        f"Found {len(valid_categories)} unique categories: {sorted(valid_categories)}")

    print("Parsing file...")
    meta_lines, projects, remaining_lines = parse_pb_file(INPUT_FILE)
    print(f"Found {len(projects)} projects to categorize")

    client = OpenAI(api_key=OPENAI_API_KEY)

    print("\nCategorizing projects...")
    for i, project in enumerate(projects, 1):
        description = project['name']
        old_category = project['category']

        print(
            f"[{i}/{len(projects)}] Processing project {project['project_id']}: {description[:60]}...")

        # If no existing category, use "uncategorized" instead of calling LLM
        if not old_category or old_category.strip() == '':
            new_category = "uncategorized"
            print(f"  No existing category, setting to 'uncategorized'")
        else:
            new_category = get_category_from_openai(
                client, description, list(valid_categories))

        project['category'] = new_category

        if old_category != new_category:
            print(f"  Updated: '{old_category}' -> '{new_category}'")
        else:
            print(f"  Category unchanged: '{new_category}'")

    print(f"\nWriting updated file: {INPUT_FILE}")
    write_pb_file(INPUT_FILE, meta_lines, projects, remaining_lines)


if __name__ == "__main__":
    main()
