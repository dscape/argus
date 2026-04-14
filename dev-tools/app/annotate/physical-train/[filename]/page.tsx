import { redirect } from "next/navigation";

type Params = Promise<{ filename: string }>;

export default async function PhysicalTrainClipPage({
  params,
}: {
  params: Params;
}) {
  const { filename } = await params;
  redirect(`/annotate/physical/${encodeURIComponent(filename)}?split=train`);
}
