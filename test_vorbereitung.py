#!/usr/bin/env python3
import chromadb

client = chromadb.PersistentClient(path='./chroma_db_store')
collection = client.get_collection('test_collection')

# Hole alle Dokumente und suche nach 'Vorbereitung'
all_docs = collection.get()
print('=== SUCHE NACH VORBEREITUNG ===')
vorbereitung_docs = [doc for doc in all_docs['documents'] if 'vorbereitung' in doc.lower()]
print(f'Gefunden: {len(vorbereitung_docs)} Dokumente mit "Vorbereitung"')
for i, doc in enumerate(vorbereitung_docs[:5]):
    print(f'{i+1}. {doc[:200]}...')
    print()

print('\n=== SUCHE NACH BANKHALTER ===')
bankhalter_docs = [doc for doc in all_docs['documents'] if 'bankhalter' in doc.lower()]
print(f'Gefunden: {len(bankhalter_docs)} Dokumente mit "Bankhalter"')
for i, doc in enumerate(bankhalter_docs[:3]):
    print(f'{i+1}. {doc[:200]}...')
    print()

print('\n=== SUCHE NACH AUFBAU/AUFBAUEN ===')
aufbau_docs = [doc for doc in all_docs['documents'] if 'aufbau' in doc.lower() or 'aufbauen' in doc.lower()]
print(f'Gefunden: {len(aufbau_docs)} Dokumente mit "Aufbau/Aufbauen"')
for i, doc in enumerate(aufbau_docs[:3]):
    print(f'{i+1}. {doc[:200]}...')
    print() 