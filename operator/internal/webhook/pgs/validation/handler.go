package validation

import (
	"context"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

//const handlerName = "podgangset-validation-webhook"

// Handler is a handler for validating PodGangSet resources.
type Handler struct {
	client client.Client
	logger logr.Logger
}

// Handle validates operations done on PodGangSet and PodGang resources.
func (h *Handler) Handle(ctx context.Context, req admission.Request) admission.Response {
	panic("implement me")
}
