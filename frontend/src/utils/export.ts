import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import { SavedArchitecture } from '@/contexts/LibraryContext';

export const exportToPDF = async (architecture: SavedArchitecture) => {
  const pdf = new jsPDF('p', 'mm', 'a4');
  const pageWidth = pdf.internal.pageSize.getWidth();
  const margin = 15;
  let yPosition = margin;

  // Title
  pdf.setFontSize(20);
  pdf.setTextColor(30, 64, 175); // Primary blue
  pdf.text(architecture.name, margin, yPosition);
  yPosition += 10;

  // Date
  pdf.setFontSize(10);
  pdf.setTextColor(100, 100, 100);
  pdf.text(`Generated: ${new Date(architecture.timestamp).toLocaleDateString()}`, margin, yPosition);
  yPosition += 10;

  // Description
  pdf.setFontSize(12);
  pdf.setTextColor(0, 0, 0);
  const descLines = pdf.splitTextToSize(architecture.description, pageWidth - 2 * margin);
  pdf.text(descLines, margin, yPosition);
  yPosition += descLines.length * 5 + 5;

  // Cost Estimate
  if (architecture.costEstimate) {
    pdf.setFontSize(14);
    pdf.setTextColor(13, 148, 136); // Accent teal
    pdf.text(`Estimated Cost: ${architecture.costEstimate}`, margin, yPosition);
    yPosition += 10;
  }

  // Services
  pdf.setFontSize(14);
  pdf.setTextColor(30, 64, 175);
  pdf.text('AWS Services:', margin, yPosition);
  yPosition += 8;

  pdf.setFontSize(11);
  pdf.setTextColor(0, 0, 0);
  architecture.services.forEach((service, index) => {
    pdf.text(`• ${service}`, margin + 5, yPosition);
    yPosition += 6;
    
    if (yPosition > 270) {
      pdf.addPage();
      yPosition = margin;
    }
  });

  // Input Requirements
  if (yPosition > 200) {
    pdf.addPage();
    yPosition = margin;
  } else {
    yPosition += 10;
  }

  pdf.setFontSize(14);
  pdf.setTextColor(30, 64, 175);
  pdf.text('Requirements:', margin, yPosition);
  yPosition += 8;

  pdf.setFontSize(10);
  pdf.setTextColor(0, 0, 0);
  const inputLines = pdf.splitTextToSize(architecture.input, pageWidth - 2 * margin);
  inputLines.forEach((line: string) => {
    if (yPosition > 280) {
      pdf.addPage();
      yPosition = margin;
    }
    pdf.text(line, margin, yPosition);
    yPosition += 5;
  });

  // Save the PDF
  pdf.save(`${architecture.name.replace(/\s+/g, '_')}_architecture.pdf`);
};

export const exportToJSON = (architecture: SavedArchitecture) => {
  const dataStr = JSON.stringify(architecture, null, 2);
  const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
  
  const exportFileDefaultName = `${architecture.name.replace(/\s+/g, '_')}_architecture.json`;
  
  const linkElement = document.createElement('a');
  linkElement.setAttribute('href', dataUri);
  linkElement.setAttribute('download', exportFileDefaultName);
  linkElement.click();
};

export const captureAndExportDiagram = async (elementId: string, fileName: string) => {
  const element = document.getElementById(elementId);
  if (!element) {
    throw new Error('Diagram element not found');
  }

  const canvas = await html2canvas(element, {
    backgroundColor: '#ffffff',
    scale: 2,
  });

  const link = document.createElement('a');
  link.download = `${fileName}_diagram.png`;
  link.href = canvas.toDataURL();
  link.click();
};
