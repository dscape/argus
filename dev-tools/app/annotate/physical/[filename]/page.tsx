import { PhysicalAnnotationPage } from "@/components/annotate/PhysicalAnnotationPage";

export default async function PhysicalClipPage({
  params,
}: {
  params: Promise<{ filename: string }>;
}) {
  const { filename } = await params;
  return <PhysicalAnnotationPage filename={decodeURIComponent(filename)} />;
}
