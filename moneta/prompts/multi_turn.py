WIKIDATA_PROMPT = """You are analyzing a Wikidata JSON entry to identify explicit mentions of economic activities.

Instructions:
- Only consider explicitly stated fields such as "industry", "product", "field of work", or equivalent.
- The keyword list below represents valid economic activity categories. Only mention an activity if there is direct evidence in the data.
- Match terms in the entry only if they clearly correspond to one of the keywords below or its direct synonym.
- When you refer to an activity, wrap it in [ ] exactly as shown in the keyword list.
- For each activity found, also mention the specific entity or value from the Wikidata entry that supports it.
- Do not invent, infer, or guess activities not explicitly stated.
- Keep the total output under 512 tokens.
- If no economic activity is present, output exactly: "No economic activity clues found."
- OUTPUT LANGUAGE MUST BE ENGLISH

Economic Activity Keywords:
- [agriculture], [forestry], [fishing], [crops], [livestock], [timber]
- [mining], [quarrying], [oil], [coal], [ores]
- [manufacturing], [processing], [assembly], [fabrication]
- [electricity], [gas], [steam]
- [water], [sewerage], [waste], [remediation]
- [construction], [building], [infrastructure]
- [wholesale], [retail], [trade], [resale], [vehicle-repair]
- [transport], [logistics], [freight], [storage], [postal]
- [accommodation], [hotels], [restaurants], [catering]
- [information], [communication], [telecom], [publishing], [IT]
- [finance], [insurance], [banking], [investment]
- [real-estate], [property], [leasing]
- [professional], [scientific], [technical], [legal], [engineering], [research]
- [administration], [support], [employment], [security], [cleaning]
- [government], [defence], [legislation], [taxation]
- [education], [training], [schooling]
- [health], [social-care], [medical], [hospitals], [clinics]
- [arts], [entertainment], [recreation], [sports], [culture]
- [personal-services], [household-services], [memberships], [repairs]
- [household-employment], [household-production]
- [extraterritorial], [embassies], [diplomacy]

Output format:
Economic activity clues:
- [keyword] — entity/value from Wikidata that supports it

JSON:
"""

WIKIPEDIA_PROMPT = """You are analyzing a Wikipedia article to identify explicit mentions of economic activities.

Instructions:
- Only consider explicitly stated fields such as "industry", "product", "field of work", or equivalent.
- The keyword list below represents valid economic activity categories. Only mention an activity if there is direct evidence in the article.
- Match terms in the article only if they clearly correspond to one of the keywords below or its direct synonym.
- When you refer to an activity, wrap it in [ ] exactly as shown in the keyword list.
- For each activity found, also mention the specific entity, phrase, or value from the article that supports it.
- Do not invent, infer, or guess activities not explicitly stated.
- Keep the total output under 512 tokens.
- If no economic activity is present, output exactly: "No economic activity clues found."
- OUTPUT LANGUAGE MUST BE ENGLISH

Economic Activity Keywords:
- [agriculture], [forestry], [fishing], [crops], [livestock], [timber]
- [mining], [quarrying], [oil], [coal], [ores]
- [manufacturing], [processing], [assembly], [fabrication]
- [electricity], [gas], [steam]
- [water], [sewerage], [waste], [remediation]
- [construction], [building], [infrastructure]
- [wholesale], [retail], [trade], [resale], [vehicle-repair]
- [transport], [logistics], [freight], [storage], [postal]
- [accommodation], [hotels], [restaurants], [catering]
- [information], [communication], [telecom], [publishing], [IT]
- [finance], [insurance], [banking], [investment]
- [real-estate], [property], [leasing]
- [professional], [scientific], [technical], [legal], [engineering], [research]
- [administration], [support], [employment], [security], [cleaning]
- [government], [defence], [legislation], [taxation]
- [education], [training], [schooling]
- [health], [social-care], [medical], [hospitals], [clinics]
- [arts], [entertainment], [recreation], [sports], [culture]
- [personal-services], [household-services], [memberships], [repairs]
- [household-employment], [household-production]
- [extraterritorial], [embassies], [diplomacy]

Output format:
Economic activity clues:
- [keyword] — entity/phrase from Wikipedia that supports it

Wikipedia content:
"""

WEBSITE_PROMPT = """You are analyzing website content to identify explicit mentions of economic activities.

Instructions:
- Only consider explicitly stated fields such as "industry", "product", "field of work", or equivalent.
- The keyword list below represents valid economic activity categories. Only mention an activity if there is direct evidence in the website content.
- Match terms in the content only if they clearly correspond to one of the keywords below or its direct synonym.
- When you refer to an activity, wrap it in [ ] exactly as shown in the keyword list.
- For each activity found, also mention the specific entity, phrase, or value from the website that supports it.
- Do not invent, infer, or guess activities not explicitly stated.
- Keep the total output under 512 tokens.
- If no economic activity is present, output exactly: "No economic activity clues found."
- OUTPUT LANGUAGE MUST BE ENGLISH

Economic Activity Keywords:
- [agriculture], [forestry], [fishing], [crops], [livestock], [timber]
- [mining], [quarrying], [oil], [coal], [ores]
- [manufacturing], [processing], [assembly], [fabrication]
- [electricity], [gas], [steam]
- [water], [sewerage], [waste], [remediation]
- [construction], [building], [infrastructure]
- [wholesale], [retail], [trade], [resale], [vehicle-repair]
- [transport], [logistics], [freight], [storage], [postal]
- [accommodation], [hotels], [restaurants], [catering]
- [information], [communication], [telecom], [publishing], [IT]
- [finance], [insurance], [banking], [investment]
- [real-estate], [property], [leasing]
- [professional], [scientific], [technical], [legal], [engineering], [research]
- [administration], [support], [employment], [security], [cleaning]
- [government], [defence], [legislation], [taxation]
- [education], [training], [schooling]
- [health], [social-care], [medical], [hospitals], [clinics]
- [arts], [entertainment], [recreation], [sports], [culture]
- [personal-services], [household-services], [memberships], [repairs]
- [household-employment], [household-production]
- [extraterritorial], [embassies], [diplomacy]

Output format:
Economic activity clues:
- [keyword] — entity/phrase from website that supports it

Website content:
"""


SATELLITE_PROMPT = """You are analyzing a satellite image description to identify explicit mentions of economic activities.

Instructions:
- Identify labeled features, POIs, or icons that clearly indicate economic activity.
- The keyword list below represents valid economic activity categories. Only mention an activity if there is direct evidence in the description.
- Match terms in the description only if they clearly correspond to one of the keywords below or its direct synonym.
- When you refer to an activity, wrap it in [ ] exactly as shown in the keyword list.
- For each activity found, also mention the specific labeled feature, POI name, or visible element that supports it.
- Do not invent, infer, or guess activities not explicitly visible or stated.
- Keep the total output under 512 tokens.
- If no economic activity is present, output exactly: "No economic activity clues found."
- OUTPUT LANGUAGE MUST BE ENGLISH

Economic Activity Keywords:
- [agriculture], [forestry], [fishing], [crops], [livestock], [timber]
- [mining], [quarrying], [oil], [coal], [ores]
- [manufacturing], [processing], [assembly], [fabrication]
- [electricity], [gas], [steam]
- [water], [sewerage], [waste], [remediation]
- [construction], [building], [infrastructure]
- [wholesale], [retail], [trade], [resale], [vehicle-repair]
- [transport], [logistics], [freight], [storage], [postal]
- [accommodation], [hotels], [restaurants], [catering]
- [information], [communication], [telecom], [publishing], [IT]
- [finance], [insurance], [banking], [investment]
- [real-estate], [property], [leasing]
- [professional], [scientific], [technical], [legal], [engineering], [research]
- [administration], [support], [employment], [security], [cleaning]
- [government], [defence], [legislation], [taxation]
- [education], [training], [schooling]
- [health], [social-care], [medical], [hospitals], [clinics]
- [arts], [entertainment], [recreation], [sports], [culture]
- [personal-services], [household-services], [memberships], [repairs]
- [household-employment], [household-production]
- [extraterritorial], [embassies], [diplomacy]

Output format:
Economic activity clues:
- [keyword] — labeled feature/POI from satellite image that supports it

Satellite image description:
"""

OSM_PROMPT = """You are analyzing an OpenStreetMap (OSM) map description to identify explicit mentions of economic activities.

Instructions:
- Identify labeled features, POIs, or tags that clearly indicate economic activity.
- The keyword list below represents valid economic activity categories. Only mention an activity if there is direct evidence in the OSM data.
- Match terms in the description only if they clearly correspond to one of the keywords below or its direct synonym.
- When you refer to an activity, wrap it in [ ] exactly as shown in the keyword list.
- For each activity found, also mention the specific OSM feature name, tag, or label that supports it.
- Do not invent, infer, or guess activities not explicitly visible or stated.
- Keep the total output under 512 tokens.
- If no economic activity is present, output exactly: "No economic activity clues found."
- OUTPUT LANGUAGE MUST BE ENGLISH

Economic Activity Keywords:
- [agriculture], [forestry], [fishing], [crops], [livestock], [timber]
- [mining], [quarrying], [oil], [coal], [ores]
- [manufacturing], [processing], [assembly], [fabrication]
- [electricity], [gas], [steam]
- [water], [sewerage], [waste], [remediation]
- [construction], [building], [infrastructure]
- [wholesale], [retail], [trade], [resale], [vehicle-repair]
- [transport], [logistics], [freight], [storage], [postal]
- [accommodation], [hotels], [restaurants], [catering]
- [information], [communication], [telecom], [publishing], [IT]
- [finance], [insurance], [banking], [investment]
- [real-estate], [property], [leasing]
- [professional], [scientific], [technical], [legal], [engineering], [research]
- [administration], [support], [employment], [security], [cleaning]
- [government], [defence], [legislation], [taxation]
- [education], [training], [schooling]
- [health], [social-care], [medical], [hospitals], [clinics]
- [arts], [entertainment], [recreation], [sports], [culture]
- [personal-services], [household-services], [memberships], [repairs]
- [household-employment], [household-production]
- [extraterritorial], [embassies], [diplomacy]

Output format:
Economic activity clues:
- [keyword] — OSM feature/tag that supports it

OSM map description:
"""


FINAL_PROMPT_NACE = """You are a helpful assistant that designed to understand economical activities from various resources.
You will be provided with clues from different sources such as Wikidata, Wikipedia, websites, OpenStreetMap (OSM) images, and satellite images.
Based on these attributes and the given name of the entity you will identify NACE Economical Activity sector code.

{NACE_CONTEXT}

Note that you may not be given all of the clues. And if you are not given any clues, you should identify based on the name of the entity.

{OUTPUT_FORMAT}
"""


MODALITY_PROMPT_MAP = {
    "osm": OSM_PROMPT,
    "satellite": SATELLITE_PROMPT,
    "wikidata": WIKIDATA_PROMPT,
    "wikipedia": WIKIPEDIA_PROMPT,
    "website": WEBSITE_PROMPT,
}
