import pandas as pd

# Sample student data (Name, Age, Subjects, and Marks)
students = [
    {"Name": "Alice", "Age": 18, "Marks": {"Math": 85, "English": 78, "Science": 92}},
    {"Name": "Bob", "Age": 17, "Marks": {"Math": 74, "English": 80, "Science": 89}},
    {"Name": "Charlie", "Age": 18, "Marks": {"Math": 90, "English": 85, "Science": 95}},
    {"Name": "David", "Age": 17, "Marks": {"Math": 68, "English": 75, "Science": 70}},
]

# Convert data into a structured list for DataFrame
processed_data = []
for student in students:
    name = student["Name"]
    age = student["Age"]
    marks = student["Marks"]
    
    # Calculate total and average marks
    total_marks = sum(marks.values())
    avg_marks = total_marks / len(marks)
    
    # Append student details with computed values
    processed_data.append({
        "Name": name,
        "Age": age,
        "Math": marks["Math"],
        "English": marks["English"],
        "Science": marks["Science"],
        "Total Marks": total_marks,
        "Average Marks": round(avg_marks, 2)
    })

# Create a DataFrame
df = pd.DataFrame(processed_data)

# Save to Excel file
excel_filename = "Student_Grades.xlsx"
df.to_excel(excel_filename, index=False)

print(f"Student data successfully saved to {excel_filename}")
