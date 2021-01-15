import spacy
from sentence_transformers import CrossEncoder


class MentionDetector:
    
    def detect(self, text):
        doc = self.model(text)
        ents = []
        types = []
        
        i_token = 0
        
        current_ent = []
        current_type = None
        
        for token in doc:
            if(token.ent_iob_=='B' and token.ent_type_): #in ent_types):
                if(len(current_ent)>0):
                    types.append(current_type)
                    ents.append(current_ent)
                
                current_type = token.ent_type_
                current_ent = []
                current_ent.append(i_token)
            elif(token.ent_iob_ == 'I' and token.ent_type_): # in ent_types):
                current_ent.append(i_token)
            
            i_token = i_token+1
            
        if(len(current_ent)>0):
            ents.append(current_ent)
            types.append(current_type)
        
        
        tokens = list(doc)
        
        return(ents, types, tokens)
    
    def context(self, text, ents, windows=3):
        doc = self.model(text)
        tokens = list(doc)
        
        window = 5
        ents_context = []

        for ent in ents:

            start_i = min(ent)
            end_i = max(ent)

            start_context = max(0, start_i-window)
            end_context = min(len(tokens), end_i+1+window)
            ents_context.append({
                    'mention': tokens[start_i:end_i+1], 
                    'context': tokens[start_context:end_context]
            })
        
        return(ents_context)
        
        
    def __init__(self, model = None):
        self.model = model
        if(self.model is None):
            self.model = spacy.load('en_core_web_lg')
            
            
            
  class Retriever:
    
    def alias_equal(mention, aliases):
    
        mention = mention.lower()
        aliases = [alias.lower() for alias in aliases]
        idx_matches = []
        for i in range(len(aliases)):
            alias = aliases[i]
            if(mention==alias):
                idx_matches.append(i)
        return(idx_matches)
    
    def retrieve(self, mention, topk=5):
        
        idx_matches = self.similarity(mention, list(self.entities['value']))
        sorted_entities = self.entities.iloc[idx_matches]
        sorted_entities = sorted_entities.drop_duplicates(keep='first', subset=['entity'])
        sorted_entities = sorted_entities[:topk]
        return(sorted_entities)
    
    
    def __init__(self, entities, similarity=None):
        self.entities = entities
        self.similarity = similarity
