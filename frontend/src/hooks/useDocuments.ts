import { useQuery } from "@tanstack/react-query";
import { apiService } from "../services/api";

// Query to get list of all documents
export function useGetDocuments() {
  return useQuery({
    queryKey: ["documents"],
    queryFn: () => apiService.getDocuments(),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

// Query to get metadata for a specific document
export function useGetDocumentMetadata(documentId: string | null) {
  return useQuery({
    queryKey: ["document-metadata", documentId],
    queryFn: () => {
      if (!documentId) return null;
      return apiService.getDocumentMetadata(documentId);
    },
    enabled: !!documentId,
  });
}

// Query to get specific page content
export function useGetDocumentPage(documentId: string | null, pageNumber: number) {
  return useQuery({
    queryKey: ["document-page", documentId, pageNumber],
    queryFn: () => {
      if (!documentId) return null;
      return apiService.getDocumentPage(documentId, pageNumber);
    },
    enabled: !!documentId && pageNumber > 0,
  });
}

// Query to get specific chunk highlight coordinates
export function useGetHighlight(documentId: string | null, chunkId: string | null) {
  return useQuery({
    queryKey: ["document-highlight", documentId, chunkId],
    queryFn: () => {
      if (!documentId || !chunkId) return null;
      return apiService.getHighlight(documentId, chunkId);
    },
    enabled: !!documentId && !!chunkId,
  });
}
