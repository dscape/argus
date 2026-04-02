import { redirect } from "next/navigation";

export default function CalibrationEvalSessionRedirect({ params }: { params: { sessionId: string } }) {
  redirect(`/evaluate/calibration/${params.sessionId}`);
}
