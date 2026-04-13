import { PhysicalAnnotationPage } from "@/components/annotate/PhysicalAnnotationPage";

export default async function PhysicalTrainClipPage({
  params,
}: {
  params: Promise<{ filename: string }>;
}) {
  const { filename } = await params;
  return <PhysicalAnnotationPage filename={decodeURIComponent(filename)} mode="train" />;
}
