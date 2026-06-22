
import re


patterns = [
    r"ignore\s+(the\s+)?(all\s+)?(previous|prior|above|earlier)\s+instructions?",
  
    r"disregard\s+(the\s+)?(all\s+)?(previous|prior|above)\s+instructions?",
    r"forget\s+(everything|all|what)\s+(you('ve| have))?\s*(been\s+told|learned|know)",

    r"you\s+are\s+now\s+(an?\s+)?(different|new|evil|unrestricted|free)",
    r"act\s+as\s+(if\s+you\s+(are|were)\s+)?(an?\s+)?(evil|unrestricted|jailbroken|DAN)",
    r"do\s+not\s+(follow|obey|respect)\s+(any\s+)?(rules?|guidelines?|restrictions?|ethics?)",

    r"your\s+new\s+(role|persona|identity|purpose)\s+is",
    r"pretend\s+(you\s+are|to\s+be)\s+",
    r"from\s+now\s+on\s+(you\s+(are|will|must|should))",

    r"(print|repeat|reveal|show|output|display)\s+(your\s+)?(system\s+prompt|instructions?|rules?|context)",
    r"what\s+(are\s+your|is\s+your)\s+(instructions?|system\s+prompt|rules?)",

    r"<\s*system\s*>",
    r"\[\s*system\s*\]",
    r"###\s*(instruction|system|prompt)",
    r"\|\|.*\|\|", 
]
compiled_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
