import { PhysicalAnnotationPage } from "@/components/annotate/PhysicalAnnotationPage";

type PhysicalAnnotationSplit = "val" | "train";

function normalizeSplit(value: string | string[] | undefined): PhysicalAnnotationSplit {
  return value === "train" ? "train" : "val";
}

export default async function PhysicalClipPage({
  params,
  searchParams,
}: {
  params: Promise<{ filename: string }>;
  searchParams: Promise<{ split?: string }>;
}) {
  const { filename } = await params;
  const resolvedSearchParams = await searchParams;
  return (
    <PhysicalAnnotationPage
      filename={decodeURIComponent(filename)}
      split={normalizeSplit(resolvedSearchParams.split)}
    />
  );
}
