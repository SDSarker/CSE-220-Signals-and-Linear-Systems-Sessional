projects = {
    "Project_A": {"priority": 1, "team_performance": 85, "deadline": 15},
    "Project_B": {"priority": 2, "team_performance": 70, "deadline": 20},
    "Project_C": {"priority": 3, "team_performance": 90, "deadline": 10},
    "Project_D": {"priority": 1, "team_performance": 60, "deadline": 25},
    "Project_E": {"priority": 3, "team_performance": 75, "deadline": 18},
    "Project_F": {"priority": 3, "team_performance": 80, "deadline": 12},
    "Project_G": {"priority": 1, "team_performance": 88, "deadline": 8},
    "Project_H": {"priority": 1, "team_performance": 65, "deadline": 30},
}


selected_projects = list ( projects . items ())[1::3]

total_resources = 0

for i , (project , details ) in enumerate(selected_projects ):
    allocated_resources = 0

    if details ["priority"] == 1:
        allocated_resources += 50
    elif details ["priority"] == 2:
        allocated_resources += 30
    else :
        allocated_resources += 20
    if details ["team_performance"] > 80:
        allocated_resources += 20
    elif details ["team_performance"] > 60:
        allocated_resources += 10
    if i % 2 == 0:
        allocated_resources += 10
    if details ["deadline"] <= 15:
        allocated_resources += 15
    else :
        allocated_resources += 5
    total_resources += allocated_resources
    print( f"Resources allocated for {project }: {allocated_resources}")



total_resources += 50


print("Total allocated resources :" , total_resources)