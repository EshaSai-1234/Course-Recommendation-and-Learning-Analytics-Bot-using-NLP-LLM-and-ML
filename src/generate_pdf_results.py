from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'NLP Course Recommendation Project - Results', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, text):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 10, text)
        self.ln()

    def add_image(self, image_path, title):
        if os.path.exists(image_path):
            self.add_page()
            self.chapter_title(title)
            # Center image
            self.image(image_path, x=10, w=190)
        else:
            print(f"Warning: Image not found {image_path}")

pdf = PDF()
pdf.add_page()

# 1. Execution Summary
pdf.chapter_title("1. Execution Summary")
summary_text = (
    "Input Type: PDF Document\n"
    "Input File: 23BCE9495 Maddela Esha Sai NLP Project Review PPT.pdf\n"
    "Status: Implementation successful with robust fallback for summarization."
)
pdf.chapter_body(summary_text)

# 2. Recommendations Text
pdf.chapter_title("2. Top Recommended Courses")
results_file = "updated_project_results.txt"
if os.path.exists(results_file):
    with open(results_file, "r", encoding="utf-8") as f:
        # Read lines and try to clean up formatting slightly
        lines = f.readlines()
        
    # Set a monospaced font for the table data
    pdf.set_font("Courier", size=8)
    for line in lines:
        try:
            # Handle potential encoding issues in FPDF (it doesn't like some utf-8 chars unless using a unicode font)
            # We'll strip non-latin-1 for safety in this basic script, or replace usage
            safe_line = line.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 5, safe_line)
        except:
            pass
else:
    pdf.chapter_body("Results file not found.")

# 3. Visualizations
pdf.add_image("updated_project_output_universities.png", "3. Top Universities")
pdf.add_image("updated_project_output_difficulty.png", "4. Difficulty Distribution")
pdf.add_image("updated_project_output_rating_diff.png", "5. Rating by Difficulty")

output_file = "NLP_Project_Results.pdf"
pdf.output(output_file)
print(f"PDF generated: {output_file}")
