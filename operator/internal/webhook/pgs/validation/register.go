package validation

import (
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

const webhookPath = "/webhooks/validate-podgangset"

// RegisterWithManager registers the webhook with the manager.
func (h *Handler) RegisterWithManager(mgr manager.Manager) error {
	webhook := &admission.Webhook{
		Handler:      h,
		RecoverPanic: ptr.To(true),
	}
	mgr.GetWebhookServer().Register(webhookPath, webhook)
	return nil
}
