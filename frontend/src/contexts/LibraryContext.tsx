import { createContext, useContext, useState, useEffect, ReactNode } from "react";

export interface SavedArchitecture {
  id: string;
  name: string;
  description: string;
  input: string;
  uploadedFile?: string;
  services: string[];
  timestamp: number;
  diagram?: string;
  costEstimate?: string;
  budgetTier?: string;
  deploymentGuide?: string;
}

interface LibraryContextType {
  architectures: SavedArchitecture[];
  saveArchitecture: (architecture: Omit<SavedArchitecture, "id" | "timestamp">) => void;
  deleteArchitecture: (id: string) => void;
  getArchitecture: (id: string) => SavedArchitecture | undefined;
}

const LibraryContext = createContext<LibraryContextType | undefined>(undefined);

export const LibraryProvider = ({ children }: { children: ReactNode }) => {
  const [architectures, setArchitectures] = useState<SavedArchitecture[]>([]);

  useEffect(() => {
    const stored = localStorage.getItem("deepcloud_library");
    if (stored) {
      setArchitectures(JSON.parse(stored));
    }
  }, []);

  const saveArchitecture = (architecture: Omit<SavedArchitecture, "id" | "timestamp">) => {
    const newArchitecture: SavedArchitecture = {
      ...architecture,
      id: Date.now().toString(),
      timestamp: Date.now(),
    };
    const updated = [...architectures, newArchitecture];
    setArchitectures(updated);
    localStorage.setItem("deepcloud_library", JSON.stringify(updated));
  };

  const deleteArchitecture = (id: string) => {
    const updated = architectures.filter((arch) => arch.id !== id);
    setArchitectures(updated);
    localStorage.setItem("deepcloud_library", JSON.stringify(updated));
  };

  const getArchitecture = (id: string) => {
    return architectures.find((arch) => arch.id === id);
  };

  return (
    <LibraryContext.Provider
      value={{
        architectures,
        saveArchitecture,
        deleteArchitecture,
        getArchitecture,
      }}
    >
      {children}
    </LibraryContext.Provider>
  );
};

export const useLibrary = () => {
  const context = useContext(LibraryContext);
  if (!context) {
    throw new Error("useLibrary must be used within LibraryProvider");
  }
  return context;
};
