import { redirect } from "next/navigation";

export default function SegmentationSessionRedirect({ params }: { params: { sessionId: string } }) {
  redirect(`/evaluate/segmentation/${params.sessionId}`);
}
