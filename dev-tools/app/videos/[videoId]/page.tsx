import { redirect } from "next/navigation";

export default async function VideoPage({ params }: { params: Promise<{ videoId: string }> }) {
  const { videoId } = await params;
  redirect(`/videos/${videoId}/info`);
}
