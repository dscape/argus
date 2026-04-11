import { ClipReviewPage } from "@/components/data/ClipReviewPage";

export default async function SyntheticClipDetailPage({
  params,
}: {
  params: Promise<{ filename: string }>;
}) {
  const { filename } = await params;
  return (
    <ClipReviewPage
      dataset="synthetic"
      directory="data/argus/train"
      filename={decodeURIComponent(filename)}
    />
  );
}
