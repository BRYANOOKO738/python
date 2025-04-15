from fpdf import FPDF

# Sample student data
students = [
    {"Name": "Alice", "Age": 18, "Marks": {"Math": 85, "English": 78, "Science": 92}},
    {"Name": "Bob", "Age": 17, "Marks": {"Math": 74, "English": 80, "Science": 89}},
    {"Name": "Charlie", "Age": 18, "Marks": {"Math": 90, "English": 85, "Science": 95}},
    {"Name": "David", "Age": 17, "Marks": {"Math": 68, "English": 75, "Science": 70}},
]

# Initialize PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", style='B', size=14)

# Title
pdf.cell(200, 10, "Student Data Report", ln=True, align='C')
pdf.ln(10)  # Line break

# Table Header
pdf.set_font("Arial", style='B', size=12)
pdf.cell(50, 10, "Name", border=1, align='C')
pdf.cell(20, 10, "Age", border=1, align='C')
pdf.cell(30, 10, "Math", border=1, align='C')
pdf.cell(30, 10, "English", border=1, align='C')
pdf.cell(30, 10, "Science", border=1, align='C')
pdf.cell(30, 10, "Average", border=1, align='C')
pdf.ln()  # New row

# Table Data
pdf.set_font("Arial", size=12)
for student in students:
    name = student["Name"]
    age = student["Age"]
    marks = student["Marks"]
    
    total_marks = sum(marks.values())
    avg_marks = total_marks / len(marks)

    pdf.cell(50, 10, name, border=1, align='C')
    pdf.cell(20, 10, str(age), border=1, align='C')
    pdf.cell(30, 10, str(marks["Math"]), border=1, align='C')
    pdf.cell(30, 10, str(marks["English"]), border=1, align='C')
    pdf.cell(30, 10, str(marks["Science"]), border=1, align='C')
    pdf.cell(30, 10, f"{avg_marks:.2f}", border=1, align='C')
    pdf.ln()  # New row

# Save PDF
pdf_filename = "Student_Report.pdf"
pdf.output(pdf_filename)
print(f"Student data successfully saved to {pdf_filename}")
