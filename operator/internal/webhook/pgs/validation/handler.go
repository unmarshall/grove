package validation

import (
	"context"
	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

const handlerName = "podgangset-validation-webhook"

type Handler struct {
	client client.Client
	logger logr.Logger
}

func (h *Handler) Handle(ctx context.Context, req admission.Request) admission.Response {
	panic("implement me")
}
