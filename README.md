# Judge-Assistant

























### Notes:
* Ranking the documents on the Document Folder by importance. 
* We may add the first documents in the Document Folder to give the Agent a context about what the case is talking about
* We need to add a node between the question classifier and retrieval node. The node is to be a metadata_retrieval that outputs either the document that the judge want itself and ends the graph or retrieve the document and retrieve something from it
