"""
constants.py
Contains constant values and mappings used across the project except prompts
"""

NACE_SECTIONS = {
    "A": "AGRICULTURE, FORESTRY AND FISHING",
    "B": "MINING AND QUARRYING",
    "C": "MANUFACTURING",
    "D": "ELECTRICITY, GAS, STEAM AND AIR CONDITIONING SUPPLY",
    "E": "WATER SUPPLY; SEWERAGE, WASTE MANAGEMENT AND REMEDIATION ACTIVITIES",
    "F": "CONSTRUCTION",
    "G": "WHOLESALE AND RETAIL TRADE",
    "H": "TRANSPORTATION AND STORAGE",
    "I": "ACCOMMODATION AND FOOD SERVICE ACTIVITIES",
    "J": "INFORMATION AND COMMUNICATION",
    "K": "FINANCIAL AND INSURANCE ACTIVITIES",
    "L": "REAL ESTATE ACTIVITIES",
    "M": "PROFESSIONAL, SCIENTIFIC AND TECHNICAL ACTIVITIES",
    "N": "ADMINISTRATIVE AND SUPPORT SERVICE ACTIVITIES",
    "O": "PUBLIC ADMINISTRATION AND DEFENCE; COMPULSORY SOCIAL SECURITY",
    "P": "EDUCATION",
    "Q": "HUMAN HEALTH AND SOCIAL WORK ACTIVITIES",
    "R": "ARTS, SPORTS AND RECREATION",
    "S": "OTHER SERVICE ACTIVITIES",
    "T": "ACTIVITIES OF HOUSEHOLDS AS EMPLOYERS AND UNDIFFERENTIATED GOODS - AND SERVICE-PRODUCING ACTIVITIES OF HOUSEHOLDS FOR OWN USE",
    "U": "ACTIVITIES OF EXTRATERRITORIAL ORGANISATIONS AND BODIES",
}

NACE_LABELS = list(NACE_SECTIONS.keys())
WEBSITE_LABELS = list("ABCDEFGHIJKLM")
NAICS2_LABELS = list("ABCDEFGHIJKLMNOPQRST")

NACE_TITLES_TO_KEYWORDS = {
    "AGRICULTURE, FORESTRY AND FISHING": [
        "agriculture",
        "forestry",
        "fishing",
        "crops",
        "livestock",
        "timber",
    ],
    "MINING AND QUARRYING": ["mining", "quarrying", "oil", "coal", "ores"],
    "MANUFACTURING": ["manufacturing", "processing", "assembly", "fabrication"],
    "ELECTRICITY, GAS, STEAM AND AIR CONDITIONING SUPPLY": [
        "electricity",
        "gas",
        "steam",
    ],
    "WATER SUPPLY; SEWERAGE, WASTE MANAGEMENT AND REMEDIATION ACTIVITIES": [
        "water",
        "sewerage",
        "waste",
        "remediation",
    ],
    "CONSTRUCTION": ["construction", "building", "infrastructure"],
    "WHOLESALE AND RETAIL TRADE": [
        "wholesale",
        "retail",
        "trade",
        "resale",
        "vehicle-repair",
    ],
    "ACCOMMODATION AND FOOD SERVICE ACTIVITIES": [
        "accommodation",
        "hotels",
        "restaurants",
        "catering",
    ],
    "TRANSPORTATION AND STORAGE": [
        "transport",
        "logistics",
        "freight",
        "storage",
        "postal",
    ],
    "INFORMATION AND COMMUNICATION": [
        "information",
        "communication",
        "telecom",
        "publishing",
        "IT",
    ],
    "FINANCIAL AND INSURANCE ACTIVITIES": [
        "finance",
        "insurance",
        "banking",
        "investment",
    ],
    "REAL ESTATE ACTIVITIES": ["real-estate", "property", "leasing"],
    "PROFESSIONAL, SCIENTIFIC AND TECHNICAL ACTIVITIES": [
        "professional",
        "scientific",
        "technical",
        "legal",
        "engineering",
        "research",
    ],
    "ADMINISTRATIVE AND SUPPORT SERVICE ACTIVITIES": [
        "administration",
        "support",
        "employment",
        "security",
        "cleaning",
    ],
    "PUBLIC ADMINISTRATION AND DEFENCE; COMPULSORY SOCIAL SECURITY": [
        "government",
        "defence",
        "legislation",
        "taxation",
    ],
    "EDUCATION": ["education", "training", "schooling"],
    "HUMAN HEALTH AND SOCIAL WORK ACTIVITIES": [
        "health",
        "social-care",
        "medical",
        "hospitals",
        "clinics",
    ],
    "ARTS, SPORTS AND RECREATION": [
        "arts",
        "entertainment",
        "recreation",
        "sports",
        "culture",
    ],
    "OTHER SERVICE ACTIVITIES": [
        "personal-services",
        "household-services",
        "memberships",
        "repairs",
    ],
    "ACTIVITIES OF HOUSEHOLDS AS EMPLOYERS AND UNDIFFERENTIATED GOODS - AND SERVICE-PRODUCING ACTIVITIES OF HOUSEHOLDS FOR OWN USE": [
        "household-employment",
        "household-production",
    ],
    "ACTIVITIES OF EXTRATERRITORIAL ORGANISATIONS AND BODIES": [
        "extraterritorial",
        "embassies",
        "diplomacy",
    ],
}

KEYWORDS_TO_NACE_TITLES = {
    keyword: category
    for category, keywords in NACE_TITLES_TO_KEYWORDS.items()
    for keyword in keywords
}
