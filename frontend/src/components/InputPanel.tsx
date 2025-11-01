import { useState, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Upload, Sparkles, X, FileText, DollarSign } from "lucide-react";
import { toast } from "sonner";
import { BudgetTier } from "@/config/api";

interface InputPanelProps {
  onGenerate: (input: string, file?: File, budgetTier?: BudgetTier) => void;
  isLoading: boolean;
  initialInput?: string;
  initialFileName?: string;
}

export const InputPanel = ({ onGenerate, isLoading, initialInput = "", initialFileName }: InputPanelProps) => {
  const [input, setInput] = useState(initialInput);
  const [dragActive, setDragActive] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [budgetTier, setBudgetTier] = useState<BudgetTier>("medium");
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (initialInput) {
      setInput(initialInput);
    }
  }, [initialInput]);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      handleFile(files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const acceptedFileTypes = [".txt", ".pdf", ".docx", ".doc"];
  const acceptedMimeTypes = [
    "text/plain",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
  ];

  const validateFile = (file: File): boolean => {
    const fileExtension = "." + file.name.split(".").pop()?.toLowerCase();
    
    if (!acceptedFileTypes.includes(fileExtension)) {
      toast.error("Invalid file type. Please upload .txt, .pdf, .docx, or .doc files.");
      return false;
    }
    
    if (file.size > 10 * 1024 * 1024) {
      toast.error("File is too large. Maximum size is 10MB.");
      return false;
    }
    
    return true;
  };

  const handleFile = (file: File) => {
    if (validateFile(file)) {
      setUploadedFile(file);
      toast.success(`File "${file.name}" uploaded successfully`);
      
      // For text files, also read content into textarea
      if (file.type === "text/plain") {
        const reader = new FileReader();
        reader.onload = (e) => {
          const text = e.target?.result as string;
          setInput((prev) => prev + (prev ? "\n\n" : "") + text);
        };
        reader.readAsText(file);
      }
    }
  };

  const clearFile = () => {
    setUploadedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    toast.info("File removed");
  };

  const handleGenerate = () => {
    if (!input.trim() && !uploadedFile) {
      toast.error("Please provide either a description or upload a file");
      return;
    }
    if (input.trim().length < 20 && !uploadedFile) {
      toast.error("Please provide more details (at least 20 characters)");
      return;
    }
    onGenerate(input, uploadedFile || undefined, budgetTier);
  };

  const charCount = input.length;
  const maxChars = 2000;

  return (
    <motion.div
      className="glass-card rounded-2xl p-6 shadow-elevated"
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
        <Sparkles className="text-accent" />
        Describe Your Application
      </h2>

      <Textarea
        placeholder="Describe your application... (e.g., 'Build a photo-sharing mobile app with user authentication, real-time notifications, image storage, and social features like comments and likes')"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        className="min-h-[200px] text-base mb-4 resize-none focus:ring-2 focus:ring-primary transition-all"
        disabled={isLoading}
      />

      <div className="flex justify-between items-center mb-4 text-sm text-muted-foreground">
        <span>
          {charCount}/{maxChars} characters
        </span>
        {charCount > maxChars && (
          <span className="text-destructive">Character limit exceeded</span>
        )}
      </div>

      {/* Budget Selector */}
      <div className="mb-6 p-4 border rounded-xl bg-muted/20">
        <div className="flex items-center gap-2 mb-3">
          <DollarSign className="w-5 h-5 text-primary" />
          <Label className="text-base font-semibold">Monthly Budget Range</Label>
        </div>
        <RadioGroup value={budgetTier} onValueChange={(value) => setBudgetTier(value as BudgetTier)}>
          <div className="flex flex-col gap-3">
            <div className="flex items-center space-x-2 p-3 rounded-lg hover:bg-muted/50 transition-colors">
              <RadioGroupItem value="low" id="low" />
              <Label htmlFor="low" className="flex-1 cursor-pointer">
                <span className="font-medium">Low</span>
                <span className="text-sm text-muted-foreground ml-2">({"<"} $100/month)</span>
              </Label>
            </div>
            <div className="flex items-center space-x-2 p-3 rounded-lg hover:bg-muted/50 transition-colors">
              <RadioGroupItem value="medium" id="medium" />
              <Label htmlFor="medium" className="flex-1 cursor-pointer">
                <span className="font-medium">Medium</span>
                <span className="text-sm text-muted-foreground ml-2">($100-$500/month)</span>
              </Label>
            </div>
            <div className="flex items-center space-x-2 p-3 rounded-lg hover:bg-muted/50 transition-colors">
              <RadioGroupItem value="high" id="high" />
              <Label htmlFor="high" className="flex-1 cursor-pointer">
                <span className="font-medium">High</span>
                <span className="text-sm text-muted-foreground ml-2">($500+/month)</span>
              </Label>
            </div>
          </div>
        </RadioGroup>
      </div>

      {/* File Upload Zone */}
      <div
        className={`border-2 border-dashed rounded-xl p-6 mb-6 text-center transition-all cursor-pointer ${
          dragActive
            ? "border-primary bg-primary/5 scale-105"
            : "border-border hover:border-primary/50 hover:bg-primary/5"
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        {uploadedFile ? (
          <div className="flex items-center justify-center gap-3">
            <FileText className="h-8 w-8 text-primary flex-shrink-0" />
            <div className="flex-1 text-left min-w-0">
              <p className="text-sm font-medium truncate">{uploadedFile.name}</p>
              <p className="text-xs text-muted-foreground">
                {(uploadedFile.size / 1024).toFixed(1)} KB
              </p>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={(e) => {
                e.stopPropagation();
                clearFile();
              }}
              disabled={isLoading}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        ) : (
          <>
            <Upload className="mx-auto mb-2 text-muted-foreground" size={32} />
            <p className="text-sm font-medium">
              Upload requirements document
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Drag & drop or click to browse
            </p>
            <p className="text-xs text-muted-foreground/70 mt-1">
              Supports .txt, .pdf, .docx, .doc (max 10MB)
            </p>
          </>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept={acceptedFileTypes.join(",")}
          className="hidden"
          onChange={handleFileInput}
          disabled={isLoading}
        />
      </div>

      <Button
        size="lg"
        className="w-full text-lg py-6 bg-gradient-to-r from-primary to-accent hover:shadow-glow transition-all duration-300 disabled:opacity-50"
        onClick={handleGenerate}
        disabled={isLoading || charCount > maxChars}
      >
        {isLoading ? (
          <>
            <motion.div
              className="mr-2 h-5 w-5 border-2 border-white border-t-transparent rounded-full"
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            />
            Analyzing requirements...
          </>
        ) : (
          <>
            <Sparkles className="mr-2" />
            Generate Architecture
          </>
        )}
      </Button>
    </motion.div>
  );
};
