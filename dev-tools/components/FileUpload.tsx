"use client";

import { useCallback, useState } from "react";
import { Upload } from "lucide-react";
import { cn } from "@/lib/utils";

interface Props {
  accept?: string;
  onFile: (file: File) => void;
  label?: string;
  className?: string;
}

export function FileUpload({
  accept,
  onFile,
  label = "Drop file here or click to browse",
  className,
}: Props) {
  const [isDragging, setIsDragging] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) {
        setFileName(file.name);
        onFile(file);
      }
    },
    [onFile]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        setFileName(file.name);
        onFile(file);
      }
    },
    [onFile]
  );

  return (
    <label
      className={cn(
        "flex flex-col items-center justify-center border-2 border-dashed rounded-lg p-8 cursor-pointer transition-colors",
        isDragging ? "border-primary bg-primary/5" : "border-muted-foreground/25 hover:border-primary/50",
        className
      )}
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
    >
      <Upload className="h-8 w-8 text-muted-foreground mb-2" />
      <span className="text-sm text-muted-foreground">{fileName || label}</span>
      <input type="file" accept={accept} onChange={handleChange} className="hidden" />
    </label>
  );
}
