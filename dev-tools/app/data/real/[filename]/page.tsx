import { ClipReviewPage } from "@/components/data/ClipReviewPage";

export default async function RealClipDetailPage({
  params,
}: {
  params: Promise<{ filename: string }>;
}) {
  const { filename } = await params;
  return (
    <ClipReviewPage
      dataset="real"
      directory="data/argus/train_real"
      filename={decodeURIComponent(filename)}
    />
  );
}
