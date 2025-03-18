import pdfplumber
import pandas as pd
import re

def extract_questions_from_pdf(pdf_path, csv_path):
    data = []
    current_question = ""
    current_options = {"A": "", "B": "", "C": "", "D": "", "E": ""}
    question_number = None
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines = text.split("\n") #separa o texto em linhas
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    
                    # Identifica início de questão utilizando Regex
                    match = re.match(r"Quest[ão]o\s*(\d+)(?:\s*[-–]?\s*(.*))?", line, re.IGNORECASE)
                    if match:
                        # Salva a questão anterior
                        if current_question.strip():
                            data.append({"Numero": question_number, "Questao": current_question.strip(), **current_options})
                        
                        question_number = int(match.group(1))
                        current_question = match.group(2) if match.group(2) else ""  # Caso haja uma descrição
                        current_options = {"A": "", "B": "", "C": "", "D": "", "E": ""}
                    
                    # Identifica alternativas (A, B, C, D, E) em linhas consecutivas
                    if re.match(r"^A\s*", lines[i]) and re.match(r"^B\s*", lines[i+1]) and re.match(r"^C\s*", lines[i+2]) and re.match(r"^D\s*", lines[i+3]) and re.match(r"^E\s*", lines[i+4]):
                        option = line[0]  # A, B, C, D ou E
                        current_options["A"] = lines[i][2:].strip()
                        current_options["B"] = lines[i+1][2:].strip()
                        current_options["C"] = lines[i+2][2:].strip()
                        current_options["D"] = lines[i+3][2:].strip()
                        current_options["E"] = lines[i+4][2:].strip()
                        i += 4
                    else:
                        # Acumula o texto da questão
                        current_question += " " + line
                    
                    i += 1
    
    # Salva última questão, caso haja
    if current_question.strip():
        data.append({"Numero": question_number, "Questao": current_question.strip(), **current_options})
    
    # Converte para DataFrame e salva em CSV
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')


extract_questions_from_pdf("prova.pdf", "questoes.csv")